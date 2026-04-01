"""
CommunityPulse-Env — Pydantic Models
All typed models for the OpenEnv spec: Observation, Action, Reward
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class NeedStatus(str, Enum):
    OPEN       = "open"
    ASSIGNED   = "assigned"
    RESOLVED   = "resolved"
    EXPIRED    = "expired"

class NeedCategory(str, Enum):
    MEDICAL    = "medical"
    FOOD       = "food"
    WATER      = "water"
    SHELTER    = "shelter"
    LOGISTICS  = "logistics"

class VolunteerSkill(str, Enum):
    MEDICAL    = "medical"
    FOOD       = "food"
    WATER      = "water"
    SHELTER    = "shelter"
    LOGISTICS  = "logistics"

class ActionType(str, Enum):
    ASSIGN      = "assign"
    WAIT        = "wait"
    INVESTIGATE = "investigate"


# ─────────────────────────────────────────────
# NEED MODEL
# ─────────────────────────────────────────────

class Need(BaseModel):
    """
    Represents one humanitarian need in the environment.
    urgency:        0.0 (low) to 1.0 (critical)
    confidence:     0.0 (very uncertain) to 1.0 (fully confirmed)
                    Represents noise from messy field data.
    people_affected: How many people this need affects.
    deadline_steps: Steps remaining before this need expires.
                    -1 means no deadline.
    required_skill: Which volunteer skill category is needed.
    status:         Current status in the environment.
    """
    id:               str
    category:         NeedCategory
    urgency:          float = Field(..., ge=0.0, le=1.0)
    confidence:       float = Field(..., ge=0.0, le=1.0)
    people_affected:  int   = Field(..., ge=1)
    deadline_steps:   int   = Field(default=-1)
    required_skill:   VolunteerSkill
    status:           NeedStatus = NeedStatus.OPEN
    assigned_volunteer: Optional[str] = None
    description:      str  = ""


# ─────────────────────────────────────────────
# VOLUNTEER MODEL
# ─────────────────────────────────────────────

class Volunteer(BaseModel):
    """
    Represents one volunteer available in the environment.
    skill:          What this volunteer specializes in.
    available:      False if currently assigned to a need.
    busy_until_step: Step number when volunteer becomes free again.
                     0 means available now.
    """
    id:               str
    name:             str
    skill:            VolunteerSkill
    available:        bool = True
    busy_until_step:  int  = 0


# ─────────────────────────────────────────────
# OBSERVATION MODEL
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """
    What the agent sees at every step.
    Returned by reset() and step().

    episode_id:      Unique ID for this episode run.
    task_id:         Which task is running (1, 2, or 3).
    step:            Current step number (starts at 0).
    time_remaining:  Steps left before episode ends.
    needs:           All needs in the current episode.
    volunteers:      All volunteers in the current episode.
    last_reward:     Reward received on the previous step.
                     0.0 on the first step after reset().
    warnings:        Environment warnings (e.g. "2 HIGH needs expiring soon").
    """
    episode_id:     str
    task_id:        int = Field(..., ge=1, le=3)
    step:           int = Field(..., ge=0)
    time_remaining: int = Field(..., ge=0)
    needs:          List[Need]
    volunteers:     List[Volunteer]
    last_reward:    float = 0.0
    warnings:       List[str] = []


# ─────────────────────────────────────────────
# ACTION MODEL
# ─────────────────────────────────────────────

class Action(BaseModel):
    """
    One atomic decision the agent makes per step.

    type:           "assign"      → assign a volunteer to a need
                    "wait"        → do nothing this step
                    "investigate" → spend a step to increase confidence
                                    on a need (useful for Task 2/3)

    need_id:        Required for "assign" and "investigate".
    volunteer_id:   Required for "assign" only.

    Examples:
        Assign:      Action(type="assign", need_id="n1", volunteer_id="v2")
        Wait:        Action(type="wait")
        Investigate: Action(type="investigate", need_id="n3")
    """
    type:          ActionType
    need_id:       Optional[str] = None
    volunteer_id:  Optional[str] = None


# ─────────────────────────────────────────────
# REWARD MODEL
# ─────────────────────────────────────────────

class Reward(BaseModel):
    """
    Feedback returned after every step().

    value:   Float reward for this step. Can be negative.
    reason:  Human-readable explanation of why this reward was given.
    done:    True if the episode is over (budget exhausted or all needs resolved).
    info:    Extra stats for logging/debugging.
             Keys: step, needs_resolved, needs_expired, volunteers_busy,
                   episode_score (final grader score, only present when done=True)
    """
    value:   float
    reason:  str
    done:    bool
    info:    dict = {}


# ─────────────────────────────────────────────
# REQUEST MODELS (for FastAPI endpoints)
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Body for POST /reset"""
    task_id: int = Field(..., ge=1, le=3)

class StepRequest(BaseModel):
    """Body for POST /step"""
    action: Action


# ─────────────────────────────────────────────
# RESPONSE MODELS (for FastAPI endpoints)
# ─────────────────────────────────────────────

class StepResponse(BaseModel):
    """Response from POST /step"""
    observation: Observation
    reward:      Reward
    done:        bool
    info:        dict = {}

class HealthResponse(BaseModel):
    """Response from GET /health"""
    status: str = "ok"
