# subagents/time_management_agent/data_models.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import datetime, time, date, time as dt_time
import uuid

# --- Data structures for storing state ---
class RecurringEvent(BaseModel):
    """Model for recurring custom habits/events"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    day_of_week: int # 0=Monday, 6=Sunday
    start_time: dt_time
    end_time: dt_time

class UnavailableTime(BaseModel):
    """Model for one-off unavailable periods"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reason: Optional[str] = None
    start_datetime: datetime
    end_datetime: datetime

# --- Input Models ---
class TopicInput(BaseModel):
    """Simplified Topic structure needed for scheduling"""
    title: str
    deadline: str # Expected format like "5 days", "1 week", "YYYY-MM-DD"
    estimated_hours: Optional[float] = Field(default=2.0)
    priority: Optional[int] = Field(default=3, ge=1, le=5, description="Priority 1-5 (1=Highest)")

class SchedulingPreferences(BaseModel):
    """User's scheduling preferences"""
    student_id: str
    preferred_days: List[int] = Field(default=[0, 1, 2, 3, 4], description="Days of week (0=Mon, 6=Sun)")
    preferred_start_time: dt_time = Field(default=dt_time(9, 0))
    preferred_end_time: dt_time = Field(default=dt_time(17, 0))
    session_duration_minutes: int = Field(default=50, ge=15)
    short_break_minutes: int = Field(default=10, ge=5)
    long_break_minutes: int = Field(default=30, ge=15)
    sessions_before_long_break: int = Field(default=4, ge=2)

class GenerateWeeklyScheduleRequest(BaseModel):
    """Request body for generating a weekly schedule"""
    student_id: str
    # Topics provided directly for this prototype endpoint
    topics: List[TopicInput]
    # Optional: week_start_date: date = Field(default_factory=date.today)

class BlockTimeRequest(BaseModel):
    """Input for the natural language block time endpoint"""
    student_id: str
    user_text: str

# --- LLM Parser Models (Internal use for parsing function) ---
class ParsedRecurringEvent(RecurringEvent): pass
class ParsedUnavailableTime(UnavailableTime): pass

# --- Output Models ---
class ScheduleEvent(BaseModel):
    """Represents one block of time in the schedule"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime
    end_time: datetime
    activity_type: str = Field(description="e.g., Study, Short Break, Long Break, Custom Habit, Unavailable")
    topic_title: Optional[str] = None # Link to topic if it's a study session
    is_recurring_habit: bool = False

class StudySchedule(BaseModel):
    """The generated schedule output"""
    student_id: str
    schedule_start_date: date
    schedule_end_date: date
    generated_at: datetime = Field(default_factory=datetime.now)
    events: List[ScheduleEvent] = Field(default=[])
    warnings: List[str] = Field(default=[], description="Potential scheduling issues")

# --- Response model for adding blocks ---
class AddBlockResponse(BaseModel):
    message: str
    event_id: str
    details: Union[RecurringEvent, UnavailableTime]