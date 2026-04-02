"""
CommunityPulse-Env — Baseline Inference Script
Runs an LLM agent against all 3 tasks and prints scores.

Requirements:
    - API_BASE_URL env var set
    - MODEL_NAME env var set
    - HF_TOKEN env var set
    - Environment server running on localhost:7860

Usage:
    python inference.py
"""

import os
import json
import time
import requests
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

MAX_INFERENCE_MINUTES = 15
START_TIME           = time.time()

# OpenAI client pointed at configured base URL
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy-key",
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def check_timeout():
    elapsed = (time.time() - START_TIME) / 60
    if elapsed > MAX_INFERENCE_MINUTES:
        print(f"\nTimeout: {MAX_INFERENCE_MINUTES} min limit reached. Stopping.")
        raise SystemExit(1)


def call_env(endpoint: str, method: str = "GET", body: dict = None) -> dict:
    """Call the environment API."""
    url = f"{ENV_URL}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=body, timeout=30)
        else:
            resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Environment call failed: {endpoint} -> {e}")
        raise


def format_observation(obs: dict) -> str:
    """Format observation into a clear prompt for the LLM."""
    lines = []
    lines.append(f"=== NGO CONTROL ROOM — Step {obs['step']} ===")
    lines.append(f"Time remaining: {obs['time_remaining']} steps")
    lines.append("")

    # Warnings
    if obs.get("warnings"):
        lines.append("⚠️  WARNINGS:")
        for w in obs["warnings"]:
            lines.append(f"   {w}")
        lines.append("")

    # Needs
    lines.append("HUMANITARIAN NEEDS:")
    for need in obs["needs"]:
        status = need["status"].upper()
        if status in ("RESOLVED", "EXPIRED"):
            continue  # skip done needs
        deadline = (
            f"DEADLINE in {need['deadline_steps']} steps"
            if need["deadline_steps"] > 0
            else "no deadline"
        )
        lines.append(
            f"  [{need['id']}] {need['category'].upper()} | "
            f"urgency={need['urgency']:.1f} | "
            f"confidence={need['confidence']:.1f} | "
            f"people={need['people_affected']} | "
            f"{deadline} | "
            f"needs_skill={need['required_skill']} | "
            f"status={need['status']}"
        )
        lines.append(f"       {need['description']}")

    lines.append("")

    # Volunteers
    lines.append("AVAILABLE VOLUNTEERS:")
    for vol in obs["volunteers"]:
        availability = (
            "AVAILABLE"
            if vol["available"]
            else f"BUSY until step {vol['busy_until_step']}"
        )
        lines.append(
            f"  [{vol['id']}] {vol['name']} | "
            f"skill={vol['skill']} | "
            f"{availability}"
        )

    lines.append("")
    lines.append(f"Last reward: {obs['last_reward']}")

    return "\n".join(lines)


def get_llm_action(obs: dict, task_id: int, retry: int = 0) -> dict:
    """
    Ask the LLM to decide the next action.
    Returns a valid action dict.
    """
    check_timeout()

    situation = format_observation(obs)

    system_prompt = """You are an AI coordinator for a humanitarian NGO.
Your job is to allocate volunteers to urgent needs as effectively as possible.

RULES:
- Prioritize HIGH urgency needs (urgency >= 0.8) first
- Match volunteer skill to need required_skill exactly when possible
- Assign before deadlines expire
- Do not assign busy volunteers
- Do not assign to already resolved/expired needs

You must respond with ONLY a valid JSON action object. Nothing else.
No explanation. No markdown. Just the JSON.

Action format:
  Assign:      {"type": "assign", "need_id": "n1", "volunteer_id": "v1"}
  Wait:        {"type": "wait"}
  Investigate: {"type": "investigate", "need_id": "n1"}
"""

    user_prompt = f"""{situation}

What is your next action? Respond with JSON only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=100,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Clean up any accidental markdown
        raw = raw.replace("```json", "").replace("```", "").strip()

        action = json.loads(raw)
        return action

    except json.JSONDecodeError as e:
        print(f"  LLM returned invalid JSON (attempt {retry+1}): {e}")
        if retry < 2:
            return get_llm_action(obs, task_id, retry + 1)
        # Fallback: wait
        return {"type": "wait"}

    except Exception as e:
        print(f"  LLM call failed (attempt {retry+1}): {e}")
        if retry < 2:
            time.sleep(2)
            return get_llm_action(obs, task_id, retry + 1)
        return {"type": "wait"}


# ─────────────────────────────────────────────
# MAIN EPISODE RUNNER
# ─────────────────────────────────────────────

def run_task(task_id: int) -> float:
    """
    Run one full episode for the given task.
    Returns the final grader score.
    """
    print(f"\n{'='*50}")
    print(f"  TASK {task_id}")
    print(f"{'='*50}")

    # Reset environment
    obs = call_env("/reset", method="POST", body={"task_id": task_id})
    print(f"Episode {obs['episode_id']} started | Budget: {obs['time_remaining']} steps")

    episode_score = 0.0
    done = False
    step = 0

    while not done:
        check_timeout()

        # Get LLM action
        action = get_llm_action(obs, task_id)
        print(f"  Step {step+1}: action={json.dumps(action)}")

        # Execute action
        result = call_env("/step", method="POST", body={"action": action})

        obs   = result["observation"]
        done  = result["done"]
        reward_val = result["reward"]["value"]
        reason     = result["reward"]["reason"]

        print(f"           reward={reward_val} | {reason[:80]}")

        if done and "episode_score" in result["info"]:
            episode_score = result["info"]["episode_score"]

        step += 1

    print(f"\n  Final score: {episode_score}")
    return episode_score


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    print("CommunityPulse-Env — Baseline Inference")
    print(f"Model:   {MODEL_NAME}")
    print(f"API URL: {API_BASE_URL}")
    print(f"Env URL: {ENV_URL}")

    # Check environment is reachable
    try:
        health = call_env("/health")
        print(f"Environment health: {health['status']}")
    except Exception:
        print("ERROR: Environment not reachable. Start the server first:")
        print("  uvicorn app.main:app --host 0.0.0.0 --port 7860")
        raise SystemExit(1)

    scores = {}

    for task_id in [1, 2, 3]:
        try:
            score = run_task(task_id)
            scores[task_id] = score
        except SystemExit:
            raise
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            scores[task_id] = 0.0

    # ── Final Results Table ──────────────────
    print(f"\n{'='*50}")
    print("  BASELINE RESULTS")
    print(f"{'='*50}")
    print(f"  Task 1 (Easy):   {scores.get(1, 0.0):.4f}")
    print(f"  Task 2 (Medium): {scores.get(2, 0.0):.4f}")
    print(f"  Task 3 (Hard):   {scores.get(3, 0.0):.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  Average:         {avg:.4f}")
    print(f"{'='*50}")

    elapsed = (time.time() - START_TIME) / 60
    print(f"\nCompleted in {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
