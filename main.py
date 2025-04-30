# subagents/time_management_agent/main.py

from fastapi import FastAPI, HTTPException, Body
from typing import Dict, List, Optional
from datetime import datetime, date # Only import needed types
import traceback

# --- Relative imports ---
try:
    from .data_models import (
        SchedulingPreferences, BlockTimeRequest, GenerateWeeklyScheduleRequest,
        StudySchedule, RecurringEvent, UnavailableTime, AddBlockResponse # Add response model
    )
except ImportError:
     from data_models import ( # Fallback
        SchedulingPreferences, BlockTimeRequest, GenerateWeeklyScheduleRequest,
        StudySchedule, RecurringEvent, UnavailableTime, AddBlockResponse
    )
try:
    from .processor import TimeProcessor
except ImportError:
     from processor import TimeProcessor # Fallback

# --- Create App and Processor Instance ---
app = FastAPI(title="Time Management & Scheduling Agent V2.2 (Refactored - Class Based)")
processor = TimeProcessor() # Instantiate the processor

# ---------------------------
# API Endpoints (Async)
# ---------------------------
@app.post("/preferences/{student_id}", status_code=200, summary="Save Student Scheduling Preferences") # Use 200 for update
async def save_preferences_endpoint(student_id: str, prefs: SchedulingPreferences = Body(...)):
    """Saves or updates scheduling preferences for a student."""
    try:
        # Ensure student_id in prefs matches path param
        if student_id != prefs.student_id:
             raise HTTPException(status_code=400, detail="Student ID in path does not match payload.")
        await processor.save_preferences(student_id, prefs)
        return {"message": "Preferences saved successfully."}
    except ValueError as e: # Catch validation errors from processor
         raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[API] Error saving preferences for {student_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save preferences: {e}")

@app.get("/preferences/{student_id}", response_model=SchedulingPreferences, summary="Get Student Scheduling Preferences")
async def get_preferences_endpoint(student_id: str):
     """Retrieves the stored scheduling preferences for the student."""
     try:
         prefs = await processor.get_preferences(student_id)
         if prefs:
             return prefs
         else:
             # If no prefs saved, should we return defaults or 404?
             # Returning 404 might be clearer API design.
             raise HTTPException(status_code=404, detail=f"Preferences not found for student {student_id}.")
     except HTTPException as e: raise e
     except Exception as e:
        print(f"[API] Error getting preferences for {student_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve preferences: {e}")

@app.post("/schedule/block-time", response_model=AddBlockResponse, status_code=201, summary="Add Habit/Block via Natural Language")
async def add_block_time_endpoint(request: BlockTimeRequest = Body(...)):
    """
    Parses natural language using LLM (via processor) to add a recurring habit
    or unavailable time block and stores it (in-memory).
    """
    if not processor.LLM_AVAILABLE: # Check processor's flag
        raise HTTPException(status_code=503, detail="LLM service unavailable for parsing.")
    try:
        student_id = request.student_id
        # Await the async parse method on the processor instance
        parsed_result = await processor.parse_block_request(request.user_text)

        if isinstance(parsed_result, RecurringEvent):
            await processor.add_recurring_habit(student_id, parsed_result) # Use processor method
            return AddBlockResponse(message="Recurring habit added.", event_id=parsed_result.event_id, details=parsed_result)
        elif isinstance(parsed_result, UnavailableTime):
            await processor.add_unavailable_time(student_id, parsed_result) # Use processor method
            return AddBlockResponse(message="Unavailable time block added.", event_id=parsed_result.event_id, details=parsed_result)
        elif isinstance(parsed_result, dict) and "error" in parsed_result:
            raise HTTPException(status_code=400, detail=f"Failed to parse request: {parsed_result['error']}")
        else:
             raise HTTPException(status_code=500, detail="Unexpected result from LLM parsing.")
    except HTTPException as e: raise e
    except Exception as e:
        print(f"[API] Error in block-time endpoint for {request.student_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not process block time request: {e}")

@app.get("/schedule/blocks/{student_id}", summary="Get Saved Habits & Blocks")
async def get_blocks_endpoint(student_id: str):
     """Gets stored recurring habits and unavailable times for the student."""
     try:
         # Call processor method to get blocks
         blocks = await processor.get_blocks(student_id)
         # Use model_dump for consistent JSON serialization
         habits_json = [RecurringEvent(**h).model_dump(mode='json') for h in blocks.get('recurring_habits', [])]
         unavailable_json = [UnavailableTime(**u).model_dump(mode='json') for u in blocks.get('unavailable_times', [])]
         return {"recurring_habits": habits_json, "unavailable_times": unavailable_json}
     except Exception as e:
        print(f"[API] Error getting blocks for {student_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve blocks: {e}")

@app.post("/schedule/generate-weekly", response_model=StudySchedule, status_code=201, summary="Generate Weekly Schedule")
async def generate_weekly_schedule_endpoint(request: GenerateWeeklyScheduleRequest = Body(...)):
    """
    Generates/Refreshes the schedule based on stored prefs/habits/blocks
    and provided topics, using the processor's logic.
    """
    student_id = request.student_id
    print(f"[API] Received weekly schedule generation request for student: {student_id}")
    try:
        # Call the synchronous generate method on the processor instance
        # No await needed here as the core logic function is sync
        generated_schedule = processor.generate_weekly_schedule(
            student_id=student_id,
            topics=request.topics,
            # Fetches prefs/blocks internally from self.storage now
            week_start_date=date.today() # Use today as start for simplicity
        )
        print(f"[API] Weekly schedule generated successfully for {student_id}")
        return generated_schedule
    except ValueError as e: # Catch specific errors like missing prefs
         raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e: raise e
    except Exception as e:
        print(f"[API] Error generation endpoint for {student_id}: {e}")
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate weekly schedule: {e}")

@app.get("/schedule/{student_id}", response_model=StudySchedule, summary="Get Latest Schedule")
async def get_schedule_endpoint(student_id: str):
    """Retrieves the latest generated schedule for the student (from memory)."""
    try:
        # Call processor method
        schedule = await processor.get_schedule(student_id)
        if schedule:
            return schedule
        else:
            raise HTTPException(status_code=404, detail=f"Schedule not found for student {student_id}.")
    except HTTPException as e: raise e
    except Exception as e:
        print(f"[API] Error getting schedule for {student_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve schedule: {e}")

@app.get("/", summary="Health Check")
async def read_root():
    """Basic health check endpoint."""
    # Access status via processor instance
    llm_status = "OK" if processor.LLM_AVAILABLE and processor.llm else "Unavailable/Error"
    return {"message": "Time Management & Scheduling Agent is running (Refactored).", "llm_parsing_status": llm_status}

# --- Cleanup Async Client on Shutdown ---
# Shutdown event remains here as it needs access to the global async_client from processor
@app.on_event("shutdown")
async def shutdown_event():
    """Closes the shared HTTPX client session on shutdown."""
    # Requires async_client to be imported from processor
    from .processor import async_client
    if async_client:
        await async_client.aclose()
        print("[API] HTTPX client closed.")

# --- Notes for Future --- now belong primarily in processor or README ---