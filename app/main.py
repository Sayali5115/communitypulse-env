"""
FastAPI server for NGO Multi-Agent Coordination Environment
Provides REST API endpoints for the RL environment
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import sys
import os

# Add parent directory to path to import the environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngo_coordination_env import NGOCoordinationEnv

app = FastAPI(
    title="NGO Multi-Agent Coordination Environment",
    description="OpenEnv-compatible RL environment for multi-agent negotiation",
    version="1.0.0"
)

# Store environment instances per session
environments = {}

class ResetRequest(BaseModel):
    session_id: str = "default"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    session_id: str = "default"
    actions: List[List[float]]  # Shape: (num_agents, 3)

class EnvConfig(BaseModel):
    num_agents: int = 3
    max_steps: int = 50

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "NGO Multi-Agent Coordination Environment",
        "version": "1.0.0",
        "description": "OpenEnv-compatible RL environment for training multi-agent negotiation strategies",
        "theme": "Multi-Agent Interactions",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/env/create": "Create new environment instance",
            "/env/reset": "Reset environment",
            "/env/step": "Take environment step",
            "/env/info": "Get environment information",
            "/results/{filename}": "Get training result files"
        },
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Environment server is running"}

@app.post("/env/create")
async def create_environment(config: EnvConfig):
    """Create a new environment instance"""
    try:
        session_id = f"env_{len(environments)}"
        env = NGOCoordinationEnv(
            num_agents=config.num_agents,
            max_steps=config.max_steps
        )
        environments[session_id] = env
        
        return {
            "session_id": session_id,
            "num_agents": config.num_agents,
            "max_steps": config.max_steps,
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/env/reset")
async def reset_environment(request: ResetRequest):
    """Reset the environment"""
    try:
        # Create environment if it doesn't exist
        if request.session_id not in environments:
            environments[request.session_id] = NGOCoordinationEnv()
        
        env = environments[request.session_id]
        observation, info = env.reset(seed=request.seed)
        
        # Convert numpy arrays and numpy scalars to Python types for JSON serialization
        obs_serializable = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                obs_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                obs_serializable[key] = value.item()
            else:
                obs_serializable[key] = value
        
        # Convert info dict as well
        info_serializable = {}
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                info_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                info_serializable[key] = value.item()
            else:
                info_serializable[key] = value
        
        return {
            "observation": obs_serializable,
            "info": info_serializable
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/env/step")
async def step_environment(request: StepRequest):
    """Take a step in the environment"""
    try:
        if request.session_id not in environments:
            raise HTTPException(status_code=404, detail="Environment not found. Call /env/reset first.")
        
        env = environments[request.session_id]
        actions = np.array(request.actions, dtype=np.float32)
        
        observation, reward, terminated, truncated, info = env.step(actions)
        
        # Convert numpy arrays and numpy scalars to Python types for JSON serialization
        obs_serializable = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                obs_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                obs_serializable[key] = value.item()
            else:
                obs_serializable[key] = value
        
        # Convert info dict as well
        info_serializable = {}
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                info_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                info_serializable[key] = value.item()
            else:
                info_serializable[key] = value
        
        return {
            "observation": obs_serializable,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info_serializable
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/env/info")
async def get_environment_info():
    """Get information about the environment"""
    env = NGOCoordinationEnv()
    
    return {
        "name": "NGO Multi-Agent Coordination Environment",
        "num_agents": env.num_agents,
        "max_steps": env.max_steps,
        "observation_space": {
            "urgency": "Discrete(10) - Urgency level 1-10",
            "available_resources": "Box(0, 100) - Volunteers available",
            "people_affected": "Box(0, 300) - People needing help",
            "other_agents_actions": "Box(0, 100) - Other agents' last actions",
            "coalition_status": "MultiBinary(3) - Coalition membership",
            "communication_channel": "Box(0, 1) - Cooperation signals"
        },
        "action_space": {
            "description": "Box(3) - [allocation%, cooperation_signal, negotiation_bid]",
            "allocation_percentage": "0-1: How much resources to deploy",
            "cooperation_signal": "0-1: Willingness to cooperate",
            "negotiation_bid": "0-1: Negotiation offer"
        },
        "tasks": [
            "Cooperation - Maximize total impact",
            "Competition - Individual performance",
            "Negotiation - Fair compromises",
            "Coalition - Strategic alliances"
        ]
    }

@app.get("/results/{filename}")
async def get_result_file(filename: str):
    """Serve training result files"""
    file_path = os.path.join("results", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@app.get("/results/")
async def list_results():
    """List available result files"""
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        return {"files": []}
    
    files = os.listdir(results_dir)
    return {
        "files": files,
        "urls": {f: f"/results/{f}" for f in files}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
