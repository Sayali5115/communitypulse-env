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
from typing import Optional
from pydantic import BaseModel

from app.env import CommunityPulseEnv
from app.graders import grade
from app.models import (
    Observation,
    Action,
    Reward,
    StepRequest,
    StepResponse,
    HealthResponse,
)
from app.env_multiagent import MultiAgentCoordinatorEnv

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
# REQUEST MODELS
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Body for POST /reset — task_id is optional, defaults to 1."""
    task_id: Optional[int] = 1


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = None):
    """
    Start a new episode. task_id defaults to 1 if not provided.
    Accepts empty body {} for validator compatibility.
    """
    if request is None:
        request = ResetRequest(task_id=1)
    task_id = request.task_id if request.task_id is not None else 1
    if task_id not in (1, 2, 3, 4):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id: {task_id}. Must be 1, 2, 3 or 4."
        )
    return env.reset(task_id)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Take one action in the environment."""
    if not env.episode_id:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first."
        )
    observation, reward, done, info = env.step(request.action)
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
    """Get current observation without consuming a step."""
    if not env.episode_id:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first."
        )
    return env.state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    return {
        "tasks": [
            {
                "id": 1,
                "name": "clean_allocation",
                "difficulty": "easy",
                "description": "Clear needs, enough volunteers. Tests basic allocation.",
                "budget": 10,
                "needs_count": 4,
                "volunteers_count": 4,
            },
            {
                "id": 2,
                "name": "prioritization_under_scarcity",
                "difficulty": "medium",
                "description": "Fewer volunteers than needs. Tests prioritization.",
                "budget": 14,
                "needs_count": 6,
                "volunteers_count": 3,
            },
            {
                "id": 3,
                "name": "deadline_skill_optimization",
                "difficulty": "hard",
                "description": "Deadlines + skill constraints + scarcity. Tests optimization.",
                "budget": 20,
                "needs_count": 10,
                "volunteers_count": 4,
            },
        ]
    }
    
# Multi-agent endpoints
multiagent_env = None

@app.post("/reset_multiagent")
async def reset_multiagent(request: dict):
    """Reset environment in multi-agent mode."""
    global multiagent_env
    
    task_id = request.get("task_id", 4)
    num_coordinators = request.get("num_coordinators", 2)
    
    # Create fresh base environment instance for multi-agent
    base_env = CommunityPulseEnv()
    
    # Wrap in multi-agent
    multiagent_env = MultiAgentCoordinatorEnv(
        base_env=base_env,
        num_coordinators=num_coordinators
    )
    
    observation = multiagent_env.reset(task_id)
    
    return {
        "observation": observation,
        "info": {"mode": "multi_agent", "coordinators": num_coordinators}
    }

@app.post("/step_multiagent")
async def step_multiagent(request: dict):
    """Take step in multi-agent environment."""
    global multiagent_env
    
    if multiagent_env is None:
        return {"error": "Environment not initialized. Call /reset_multiagent first"}
    
    action = request.get("action")
    coordinator_id = request.get("coordinator_id")
    
    result = multiagent_env.step(action, coordinator_id)
    
    return result

@app.get("/state_multiagent")
async def get_state_multiagent():
    """Get current multi-agent state."""
    global multiagent_env
    
    if multiagent_env is None:
        return {"error": "Environment not initialized"}
    
    base_obs = multiagent_env.base_env.get_state()
    return multiagent_env._get_multiagent_observation(base_obs)

@app.get("/leaderboard")
async def get_leaderboard():
    """Get coordinator rankings."""
    global multiagent_env
    
    if multiagent_env is None:
        return {"error": "Multi-agent environment not initialized"}
    
    return {"leaderboard": multiagent_env.get_leaderboard()}
