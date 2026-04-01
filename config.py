"""
CommunityPulse-Env — Central Configuration
All environment variables are read here.
Every other file imports from this module.
Never hardcode these values anywhere else.
"""

import os

# ─────────────────────────────────────────────
# LLM CONFIG (required for inference.py)
# ─────────────────────────────────────────────

# The API endpoint for the LLM (e.g. https://api.openai.com/v1)
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")

# The model identifier to use for inference (e.g. gpt-4o)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

# Hugging Face / API key
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─────────────────────────────────────────────
# ENVIRONMENT CONFIG
# ─────────────────────────────────────────────

# Step budgets per task
TASK_BUDGETS = {
    1: 10,   # Easy — enough steps to assign all needs
    2: 14,   # Medium — tight but manageable
    3: 20,   # Hard — requires optimal decisions
}

# Number of needs per task
TASK_NEED_COUNTS = {
    1: 4,
    2: 6,
    3: 10,
}

# Number of volunteers per task
TASK_VOLUNTEER_COUNTS = {
    1: 4,   # Equal to needs — no scarcity
    2: 3,   # Fewer than needs — must prioritize
    3: 4,   # Much fewer than needs — must optimize
}

# ─────────────────────────────────────────────
# REWARD VALUES
# ─────────────────────────────────────────────

REWARD_CORRECT_ASSIGN       =  1.0   # right skill, right need
REWARD_URGENCY_BONUS        =  0.5   # extra for resolving HIGH urgency need
REWARD_DEADLINE_BONUS       =  0.5   # extra for resolving before deadline
REWARD_INVESTIGATE_GAIN     =  0.2   # small reward for useful investigate

PENALTY_WRONG_SKILL         = -0.5   # assigned wrong skill
PENALTY_UNAVAILABLE_VOL     = -0.3   # tried to assign busy volunteer
PENALTY_ALREADY_RESOLVED    = -0.2   # tried to assign already resolved need
PENALTY_EXPIRED_NEED        = -2.0   # HIGH urgency need expired unassigned
PENALTY_LOOP                = -2.0   # repeated same action twice
PENALTY_LAZY_WAIT           = -0.3   # waited when volunteers were available

# ─────────────────────────────────────────────
# SERVER CONFIG
# ─────────────────────────────────────────────

HOST = "0.0.0.0"
PORT = 7860   # HF Spaces default port

# ─────────────────────────────────────────────
# INFERENCE CONFIG
# ─────────────────────────────────────────────

MAX_INFERENCE_MINUTES = 15   # hard timeout for inference.py
MAX_RETRIES           = 3    # LLM call retries on failure
