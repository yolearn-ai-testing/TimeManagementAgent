# subagents/time_management_agent/processor.py

import os
import uuid
import json
from typing import List, Optional, Dict, Union
from datetime import datetime, time, timedelta, date, time as dt_time
from collections import defaultdict
from dotenv import load_dotenv
import traceback
import asyncio

# --- Check and Import LLM Libraries ---
LANGCHAIN_AVAILABLE_FLAG = False
llm_client_class = None
prompt_template_class = None
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    llm_client_class = ChatGoogleGenerativeAI
    prompt_template_class = PromptTemplate
    LANGCHAIN_AVAILABLE_FLAG = True
except ImportError:
    print("WARNING: Langchain or Google GenAI not installed. LLM features will be disabled.")

# --- Relative Import for Models ---
try:
    from .data_models import (
        RecurringEvent, UnavailableTime, TopicInput, SchedulingPreferences,
        ScheduleEvent, StudySchedule, ParsedRecurringEvent, ParsedUnavailableTime
    )
except ImportError:
    from data_models import ( # Fallback
        RecurringEvent, UnavailableTime, TopicInput, SchedulingPreferences,
        ScheduleEvent, StudySchedule, ParsedRecurringEvent, ParsedUnavailableTime
    )

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Use getenv

class TimeProcessor:
    """Encapsulates logic and data storage for the Time Management Agent."""

    def __init__(self):
        """Initializes storage and the LLM client if configured."""
        # --- In-memory storage (Managed by this processor instance) ---
        self.schedules: Dict[str, StudySchedule] = {}
        self.preferences: Dict[str, SchedulingPreferences] = {}
        self.recurring_habits_db: Dict[str, List[RecurringEvent]] = defaultdict(list)
        self.unavailable_times_db: Dict[str, List[UnavailableTime]] = defaultdict(list)

        # --- LLM Setup ---
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.block_time_prompt_template: Optional[PromptTemplate] = None
        self.LLM_AVAILABLE: bool = False

        if LANGCHAIN_AVAILABLE_FLAG and GOOGLE_API_KEY and llm_client_class and prompt_template_class:
            try:
                self.llm = llm_client_class(
                    model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3
                )
                self.block_time_prompt_template = prompt_template_class(
                    template="""
Parse the user's request to block out time. Identify if it's a single unavailable period or a recurring event/habit.
Today's date is {today_date}. Assume the current year unless specified otherwise. If possible infer year from context if provided (e.g. 'next thursday').

User Request: "{user_text}"

Determine the type ('unavailable' or 'recurring') and extract the relevant details.
For 'unavailable', extract: start_datetime (ISO format string YYYY-MM-DDTHH:MM:SS), end_datetime (ISO format string YYYY-MM-DDTHH:MM:SS), reason (optional string). Handle date ranges like 'this weekend', 'next Mon-Wed'.
For 'recurring', extract: title (string, e.g., 'Gym', 'Piano Lesson'), day_of_week (integer, Monday=0, Sunday=6), start_time (HH:MM:SS), end_time (HH:MM:SS). Handle terms like 'every weekday', 'weekends'.

Output ONLY a single JSON object containing 'event_type' (either 'unavailable' or 'recurring') and a 'data' field containing the extracted details matching the type.

Example unavailable: {{"event_type": "unavailable", "data": {{"reason": "Doctor Appointment", "start_datetime": "2025-04-15T11:00:00", "end_datetime": "2025-04-15T12:00:00"}}}}
Example recurring: {{"event_type": "recurring", "data": {{"title": "Guitar Class", "day_of_week": 2, "start_time": "18:00:00", "end_time": "19:00:00"}}}}
Example error: {{"event_type": "error", "data": {{"message": "Could not parse the request clearly."}}}}

JSON Output:
""",
                    input_variables=["user_text", "today_date"],
                )
                self.LLM_AVAILABLE = True
                print("TimeProcessor: LLM Initialized for parsing block time.")
            except Exception as e:
                print(f"TimeProcessor ERROR: Failed LLM init: {e}")
                self.LLM_AVAILABLE = False
                self.llm = None
        else:
            if not LANGCHAIN_AVAILABLE_FLAG: print("TimeProcessor WARNING: LLM libraries not installed.")
            if not GOOGLE_API_KEY: print("TimeProcessor WARNING: GOOGLE_API_KEY missing.")
            self.LLM_AVAILABLE = False
            print("TimeProcessor WARNING: LLM parsing disabled.")

    # --- Helper Methods (Synchronous) ---
    def _parse_deadline(self, deadline_str: str, today: date) -> Optional[date]:
        # (Same logic as before)
        try:
            dl_str = deadline_str.lower() # Use local var
            if "days" in dl_str: return today + timedelta(days=int(dl_str.split()[0]))
            if "week" in dl_str: return today + timedelta(weeks=int(dl_str.split()[0]))
            return date.fromisoformat(dl_str)
        except ValueError:
             for fmt in ("%Y/%m/%d", "%d-%m-%y", "%d/%m/%y", "%d-%b-%y", "%d %b %y"):
                try: return datetime.strptime(dl_str, fmt).date()
                except ValueError: continue
             print(f"Warning: Could not parse deadline '{deadline_str}'")
             return None
        except Exception as e: print(f"Warning: Error parsing deadline '{deadline_str}': {e}"); return None

    def _get_blocked_intervals(self, start_date: date, end_date: date, prefs: SchedulingPreferences, student_recurring_habits: List[RecurringEvent], student_unavailable_times: List[UnavailableTime]) -> List[tuple[datetime, datetime]]:
        # (Same logic as before - calculates all non-study intervals)
        # ... (Full logic omitted for brevity - ensure it's copied from previous version) ...
        blocked = []; current_d = start_date
        while current_d <= end_date:
            day_idx = current_d.weekday()
            for habit in student_recurring_habits:
                if habit.day_of_week == day_idx:
                    s_dt=datetime.combine(current_d, habit.start_time); e_dt=datetime.combine(current_d, habit.end_time)
                    if e_dt < s_dt: e_dt += timedelta(days=1)
                    if s_dt.date() <= end_date and e_dt.date() >= start_date: blocked.append((s_dt, e_dt))
            day_start=datetime.combine(current_d, dt_time.min); pref_start=datetime.combine(current_d, prefs.preferred_start_time); pref_end=datetime.combine(current_d, prefs.preferred_end_time); day_end=datetime.combine(current_d, dt_time.max)
            if pref_start > day_start: blocked.append((day_start, pref_start))
            if pref_end < day_end: blocked.append((pref_end, day_end))
            if day_idx not in prefs.preferred_days and pref_start < pref_end: blocked.append((pref_start, pref_end))
            current_d += timedelta(days=1)
        for unavailable in student_unavailable_times:
            if unavailable.start_datetime.date() <= end_date and unavailable.end_datetime.date() >= start_date: blocked.append((unavailable.start_datetime, unavailable.end_datetime))
        if not blocked: return []
        blocked.sort(); merged = []; current_start, current_end = blocked[0]
        for next_start, next_end in blocked[1:]:
            if next_start < current_end: current_end = max(current_end, next_end)
            else: merged.append((current_start, current_end)); current_start, current_end = next_start, next_end
        merged.append((current_start, current_end)); return merged


    # --- Core Logic Methods ---
    # Needs to be async because it calls LLM
    async def parse_block_request(self, user_text: str) -> Union[RecurringEvent, UnavailableTime, Dict]:
        """Uses LLM to parse natural language into structured block time (Async)."""
        if not self.LLM_AVAILABLE or not self.llm or not self.block_time_prompt_template:
            print("[Processor] LLM parsing called but LLM unavailable.")
            return {"error": "LLM service not available for parsing."}

        today_str = date.today().isoformat()
        prompt = self.block_time_prompt_template.format(user_text=user_text, today_date=today_str)
        try:
            response = await self.llm.ainvoke(prompt) # Use self.llm
            content = response.content.strip()
            print(f"[Processor] LLM Raw Output for Parsing:\n{content}")
            if content.startswith("```json"): content = content[7:-3].strip()
            elif content.startswith("```"): content = content[3:-3].strip()
            try:
                 parsed_json = json.loads(content)
                 event_type = parsed_json.get("event_type"); data = parsed_json.get("data")
                 if event_type == "recurring" and data:
                     if all(k in data for k in ['title', 'day_of_week', 'start_time', 'end_time']):
                          data.setdefault('event_id', str(uuid.uuid4())); return RecurringEvent(**data)
                     else: return {"error": "LLM missed required recurring event fields."}
                 elif event_type == "unavailable" and data:
                      if all(k in data for k in ['start_datetime', 'end_datetime']):
                           data.setdefault('event_id', str(uuid.uuid4()))
                           data['start_datetime'] = datetime.fromisoformat(data['start_datetime'])
                           data['end_datetime'] = datetime.fromisoformat(data['end_datetime'])
                           return UnavailableTime(**data)
                      else: return {"error": "LLM missed required unavailable time fields."}
                 elif event_type == "error" and data: return {"error": data.get("message", "LLM parsing failed.")}
                 else: return {"error": "LLM returned unexpected format."}
            except json.JSONDecodeError as e: return {"error": f"LLM returned invalid JSON: {e}"}
            except Exception as parse_e: return {"error": f"LLM output failed validation: {parse_e}"}
        except Exception as e:
            print(f"ERROR processing LLM block time request: {e}")
            return {"error": f"Unexpected error during LLM parsing: {e}"}

    # This core scheduling logic remains synchronous as it's CPU-bound
    def generate_weekly_schedule(self, student_id: str, topics: List[TopicInput], week_start_date: date = date.today()) -> StudySchedule:
        """Generates the study schedule for the week (Sync Logic)."""
        print(f"[Processor] Generating weekly schedule for {student_id} starting {week_start_date}")
        # Fetch needed data from instance storage
        prefs = self.preferences.get(student_id)
        if not prefs: raise ValueError(f"Preferences not found for student {student_id}.") # Raise internal error
        recurring_habits = self.recurring_habits_db.get(student_id, [])
        unavailable_times = self.unavailable_times_db.get(student_id, [])

        events: List[ScheduleEvent] = []
        warnings: List[str] = []
        schedule_end_date = week_start_date + timedelta(days=6)
        today = date.today()

        # Process Topics (Sync)
        parsed_topics = []
        for topic in topics:
            deadline_date = self._parse_deadline(topic.deadline, today) # Use helper method
            if deadline_date: parsed_topics.append({ "title": topic.title, "deadline": deadline_date, "minutes_needed": int(topic.estimated_hours * 60), "priority": topic.priority })
            else: warnings.append(f"Could not parse deadline for '{topic.title}'.")
        parsed_topics.sort(key=lambda x: (x["priority"], x["deadline"]))

        # Calculate Blocks & Add Fixed Events (Sync)
        blocked_intervals = self._get_blocked_intervals(week_start_date, schedule_end_date, prefs, recurring_habits, unavailable_times) # Use helper
        # Add habit/unavailable events (Sync loop - same as before)
        # ... (Full logic omitted for brevity - ensure it's copied)...
        for habit in recurring_habits:
             current_d = week_start_date
             while current_d <= schedule_end_date:
                 if current_d.weekday() == habit.day_of_week:
                     s_dt = datetime.combine(current_d, habit.start_time); e_dt = datetime.combine(current_d, habit.end_time)
                     if e_dt < s_dt: e_dt += timedelta(days=1)
                     if s_dt.date() == current_d and e_dt.date() >= current_d:
                          events.append(ScheduleEvent(event_id=habit.event_id, start_time=s_dt, end_time=e_dt, activity_type="Custom Habit", topic_title=habit.title, is_recurring_habit=True))
                 current_d += timedelta(days=1)
        for unavailable in unavailable_times:
             start = max(unavailable.start_datetime, datetime.combine(week_start_date, dt_time.min)); end = min(unavailable.end_datetime, datetime.combine(schedule_end_date, dt_time.max))
             if start < end: events.append(ScheduleEvent(event_id=unavailable.event_id, start_time=start, end_time=end, activity_type="Unavailable", topic_title=unavailable.reason))


        # Fill Remaining Time (Sync Logic Loop - same as before)
        # ... (Full scheduling loop logic omitted for brevity - ensure it's copied)...
        topic_minutes_scheduled: Dict[str, int] = defaultdict(int); current_datetime_cursor = datetime.combine(week_start_date, dt_time.min); end_of_schedule_dt = datetime.combine(schedule_end_date, dt_time.max)
        remaining_topics_list = [t for t in parsed_topics]; topic_idx = 0; session_in_block = 0
        while current_datetime_cursor < end_of_schedule_dt and topic_idx < len(remaining_topics_list):
             topic_info = remaining_topics_list[topic_idx]; topic_title = topic_info["title"]; minutes_still_needed = topic_info["minutes_needed"] - topic_minutes_scheduled[topic_title]; topic_deadline = topic_info["deadline"]
             if minutes_still_needed <= 0: topic_idx += 1; continue
             potential_start = current_datetime_cursor; slot_found = False
             while potential_start < end_of_schedule_dt:
                  potential_end = potential_start + timedelta(minutes=prefs.session_duration_minutes)
                  is_pref_day = potential_start.weekday() in prefs.preferred_days
                  is_pref_time = prefs.preferred_start_time <= potential_start.time() < prefs.preferred_end_time and potential_end.time() <= prefs.preferred_end_time and potential_start.date() == potential_end.date()
                  if not (is_pref_day and is_pref_time):
                       next_pref_start_today = datetime.combine(potential_start.date(), prefs.preferred_start_time);
                       if potential_start < next_pref_start_today: potential_start = next_pref_start_today
                       else: potential_start = datetime.combine(potential_start.date() + timedelta(days=1), prefs.preferred_start_time)
                       current_datetime_cursor = potential_start; continue
                  is_blocked = False
                  for block_start, block_end in blocked_intervals:
                      if potential_start < block_end and potential_end > block_start: potential_start = block_end; is_blocked = True; break
                  if is_blocked: current_datetime_cursor = potential_start; continue
                  if potential_start.date() > topic_deadline: warnings.append(f"Could not schedule remaining time for '{topic_title}' before deadline {topic_deadline}."); topic_minutes_scheduled[topic_title] = topic_info["minutes_needed"]; slot_found = False; break
                  slot_found = True; current_datetime_cursor = potential_start; break
             if not slot_found:
                  if potential_start >= end_of_schedule_dt and minutes_still_needed > 0 : warnings.append(f"Ran out of time in week for '{topic_title}'.")
                  topic_idx += 1; continue
             session_end = current_datetime_cursor + timedelta(minutes=prefs.session_duration_minutes)
             events.append(ScheduleEvent(start_time=current_datetime_cursor, end_time=session_end, activity_type="Study", topic_title=topic_title))
             topic_minutes_scheduled[topic_title] += prefs.session_duration_minutes; minutes_still_needed -= prefs.session_duration_minutes; session_in_block += 1
             is_long_break = (session_in_block % prefs.sessions_before_long_break == 0); break_duration = prefs.long_break_minutes if is_long_break else prefs.short_break_minutes
             break_start = session_end; break_end = break_start + timedelta(minutes=break_duration)
             is_break_blocked = False; is_break_past_pref_end = break_end.time() > prefs.preferred_end_time or break_end.date() > current_datetime_cursor.date()
             for block_start, block_end in blocked_intervals:
                 if break_start < block_end and break_end > block_start: is_break_blocked = True; break
             if not is_break_blocked and not is_break_past_pref_end: events.append(ScheduleEvent(start_time=break_start, end_time=break_end, activity_type="Long Break" if is_long_break else "Short Break")); current_datetime_cursor = break_end
             else: current_datetime_cursor = session_end
             if minutes_still_needed <= 0: topic_idx += 1


        # Final Processing (Sync)
        for topic_info in parsed_topics:
             if topic_minutes_scheduled[topic_info["title"]] < topic_info["minutes_needed"]:
                 mins_missed = topic_info["minutes_needed"] - topic_minutes_scheduled[topic_info["title"]]
                 warnings.append(f"Could not schedule all estimated time ({mins_missed} mins left) for '{topic_info['title']}'.")
        events.sort(key=lambda x: x.start_time)

        # Store generated schedule
        new_schedule = StudySchedule(
            student_id=student_id, schedule_start_date=week_start_date, schedule_end_date=schedule_end_date,
            events=events, warnings=warnings
        )
        self.schedules[student_id] = new_schedule # Save to instance storage
        return new_schedule

    # --- Methods for managing stored data ---
    async def save_preferences(self, student_id: str, prefs: SchedulingPreferences) -> bool:
         if student_id != prefs.student_id: raise ValueError("Student ID mismatch")
         self.preferences[student_id] = prefs
         return True

    async def get_preferences(self, student_id: str) -> Optional[SchedulingPreferences]:
         return self.preferences.get(student_id)

    async def add_recurring_habit(self, student_id: str, habit: RecurringEvent):
        self.recurring_habits_db[student_id].append(habit)

    async def add_unavailable_time(self, student_id: str, unavailable: UnavailableTime):
        self.unavailable_times_db[student_id].append(unavailable)

    async def get_blocks(self, student_id: str) -> Dict:
        return {
            "recurring_habits": self.recurring_habits_db.get(student_id, []),
            "unavailable_times": self.unavailable_times_db.get(student_id, [])
         }

    async def get_schedule(self, student_id: str) -> Optional[StudySchedule]:
         return self.schedules.get(student_id)