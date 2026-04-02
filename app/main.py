"""
CommunityPulse-Env — FastAPI Application
Exposes the environment as HTTP endpoints per OpenEnv spec.

Endpoints:
    GET  /health  → health check
    POST /reset   → start new episode
    POST /step    → take one action
    GET  /state   → current observation (no step consumed)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.env import CommunityPulseEnv
from app.graders import grade
from app.models import (
    Observation,
    Action,
    Reward,
    ResetRequest,
    StepRequest,
    StepResponse,
    HealthResponse,
)

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="CommunityPulse-Env",
    description=(
        "An RL environment simulating humanitarian NGO volunteer coordination. "
        "An agent allocates limited volunteers to urgent needs "
        "under time pressure and resource scarcity."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# GLOBAL ENVIRONMENT INSTANCE
# ─────────────────────────────────────────────

env = CommunityPulseEnv()


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint.
    Returns 200 + {"status": "ok"} if the service is running.
    Required for HF Space automated ping.
    """
    return HealthResponse(status="ok")


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest):
    """
    Start a new episode for the given task.

    Body:
        { "task_id": 1 | 2 | 3 }

    Returns:
        Initial Observation with all needs and volunteers.
    """
    if request.task_id not in (1, 2, 3):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id: {request.task_id}. Must be 1, 2, or 3."
        )
    observation = env.reset(request.task_id)
    return observation


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Take one action in the environment.

    Body:
        {
            "action": {
                "type": "assign" | "wait" | "investigate",
                "need_id": "n1",         (required for assign/investigate)
                "volunteer_id": "v2"     (required for assign)
            }
        }

    Returns:
        observation: updated environment state
        reward:      reward value + reason + done flag
        done:        True if episode is over
        info:        episode stats

    If done=True, the grader score is included in info["episode_score"].
    """
    if not env.episode_id:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first."
        )

    observation, reward, done, info = env.step(request.action)

    # If episode is done, run grader and attach score
    if done:
        grader_result = grade(env, env.task_id)
        info["episode_score"] = grader_result["score"]
        info["grader_detail"] = grader_result
        reward.info = info

    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=Observation)
def state():
    """
    Get the current observation without consuming a step.

    Returns:
        Current Observation (same as last step's observation).
    """
    if not env.episode_id:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first."
        )
    return env.state()


@app.get("/tasks")
def list_tasks():
    """
    List all available tasks with descriptions.
    Useful for the OpenEnv validator.
    """
    return {
        "tasks": [
            {
                "id": 1,
                "name": "clean_allocation",
                "difficulty": "easy",
                "description": (
                    "Clear needs, enough volunteers. "
                    "Agent must assign correctly. "
                    "Tests basic allocation capability."
                ),
                "budget": 10,
                "needs_count": 4,
                "volunteers_count": 4,
            },
            {
                "id": 2,
                "name": "prioritization_under_scarcity",
                "difficulty": "medium",
                "description": (
                    "Some uncertainty in urgency, fewer volunteers than needs. "
                    "Agent must prioritize HIGH urgency needs. "
                    "Tests prioritization and resource allocation."
                ),
                "budget": 14,
                "needs_count": 6,
                "volunteers_count": 3,
            },
            {
                "id": 3,
                "name": "deadline_skill_optimization",
                "difficulty": "hard",
                "description": (
                    "Many needs, few volunteers, tight deadlines, skill constraints. "
                    "Agent must optimize coverage, skill match, and deadline adherence. "
                    "Tests multi-objective optimization under pressure."
                ),
                "budget": 20,
                "needs_count": 10,
                "volunteers_count": 4,
            },
        ]
    }
