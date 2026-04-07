"""
CommunityPulse-Env — Baseline Inference Script
Runs an LLM agent against all 3 tasks and prints scores.

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Environment variables:
    API_BASE_URL   — LLM API endpoint
    MODEL_NAME     — model identifier
    HF_TOKEN       — HuggingFace / API key (preferred)
    OPENAI_API_KEY — OpenAI API key (fallback)
    ENV_URL        — environment server URL (default: http://localhost:7860)

Usage:
    python inference.py
"""

import os
import json
import time
import requests
from typing import Optional, List
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_BASE_URL   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN       = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
API_KEY        = HF_TOKEN or OPENAI_API_KEY   # HF_TOKEN takes priority

ENV_URL        = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK             = "communitypulse-env"
MAX_INFERENCE_MINUTES = 15
START_TIME            = time.time()
SUCCESS_THRESHOLD     = 0.5

# Fail early if no API key at all
if not API_KEY:
    print("[DEBUG] WARNING: No API key found. Set HF_TOKEN or OPENAI_API_KEY.", flush=True)

# OpenAI client — works with any OpenAI-compatible endpoint
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY if API_KEY else "dummy-key",
)


# ─────────────────────────────────────────────
# MANDATORY LOG FUNCTIONS
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Sanitize error — remove newlines to keep output on single line
    if error:
        error_val = error.replace("\n", " ").replace("\r", " ").strip()
    else:
        error_val = "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def check_timeout():
    elapsed = (time.time() - START_TIME) / 60
    if elapsed > MAX_INFERENCE_MINUTES:
        print(f"[DEBUG] Timeout: {MAX_INFERENCE_MINUTES} min limit reached.", flush=True)
        raise SystemExit(1)


def call_env(endpoint: str, method: str = "GET", body: dict = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=body or {}, timeout=30)
        else:
            resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[DEBUG] Environment call failed: {endpoint} -> {e}", flush=True)
        raise


def format_observation(obs: dict) -> str:
    lines = []
    lines.append(f"=== NGO CONTROL ROOM — Step {obs['step']} ===")
    lines.append(f"Time remaining: {obs['time_remaining']} steps")

    if obs.get("warnings"):
        lines.append("WARNINGS:")
        for w in obs["warnings"]:
            lines.append(f"  {w}")

    lines.append("\nHUMANITARIAN NEEDS:")
    for need in obs["needs"]:
        if need["status"] in ("resolved", "expired"):
            continue
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
            f"{deadline} | needs_skill={need['required_skill']}"
        )

    lines.append("\nAVAILABLE VOLUNTEERS:")
    for vol in obs["volunteers"]:
        status = "AVAILABLE" if vol["available"] else f"BUSY until step {vol['busy_until_step']}"
        lines.append(f"  [{vol['id']}] {vol['name']} | skill={vol['skill']} | {status}")

    return "\n".join(lines)


def get_llm_action(obs: dict, retry: int = 0) -> dict:
    check_timeout()

    system_prompt = """You are an AI coordinator for a humanitarian NGO.
Allocate volunteers to urgent needs as effectively as possible.

RULES:
- Prioritize HIGH urgency needs (urgency >= 0.8) first
- Match volunteer skill to need required_skill exactly
- Assign before deadlines expire
- Do not assign busy volunteers
- Do not assign to resolved/expired needs

Respond with ONLY a valid JSON action. No explanation. No markdown.

Action format:
  Assign:      {"type": "assign", "need_id": "n1", "volunteer_id": "v1"}
  Wait:        {"type": "wait"}
  Investigate: {"type": "investigate", "need_id": "n1"}"""

    user_prompt = format_observation(obs) + "\n\nWhat is your next action? JSON only."

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
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError:
        if retry < 2:
            return get_llm_action(obs, retry + 1)
        return {"type": "wait"}

    except Exception as e:
        print(f"[DEBUG] LLM call failed (attempt {retry+1}): {e}", flush=True)
        if retry < 2:
            time.sleep(2)
            return get_llm_action(obs, retry + 1)
        return {"type": "wait"}


# ─────────────────────────────────────────────
# TASK RUNNER
# ─────────────────────────────────────────────

TASK_NAMES = {
    1: "clean_allocation",
    2: "prioritization_under_scarcity",
    3: "deadline_skill_optimization",
}


def run_task(task_id: int) -> float:
    task_name = TASK_NAMES[task_id]
    rewards:  List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs  = call_env("/reset", method="POST", body={"task_id": task_id})
        done = False

        for step in range(1, 200):
            check_timeout()
            if done:
                break

            action     = get_llm_action(obs)
            action_str = json.dumps(action, separators=(',', ':'))
            error      = None

            try:
                result = call_env("/step", method="POST", body={"action": action})
                obs    = result["observation"]
                done   = result["done"]
                reward = result["reward"]["value"]
            except Exception as e:
                reward = 0.0
                done   = True
                error  = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                try:
                    score = result["info"].get("episode_score", 0.0)
                except Exception:
                    score = 0.0
                break

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        success = False
        score   = 0.0

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"[DEBUG] Model:   {MODEL_NAME}", flush=True)
    print(f"[DEBUG] API URL: {API_BASE_URL}", flush=True)
    print(f"[DEBUG] Env URL: {ENV_URL}", flush=True)
    print(f"[DEBUG] API Key: {'set' if API_KEY else 'NOT SET'}", flush=True)

    # Check environment is reachable
    try:
        health = call_env("/health")
        print(f"[DEBUG] Environment health: {health['status']}", flush=True)
    except Exception:
        print("[DEBUG] ERROR: Environment not reachable. Start server first:", flush=True)
        print("[DEBUG]   uvicorn app.main:app --host 0.0.0.0 --port 7860", flush=True)
        raise SystemExit(1)

    scores = {}

    for task_id in [1, 2, 3]:
        try:
            score = run_task(task_id)
            scores[task_id] = score
        except SystemExit:
            raise
        except Exception as e:
            print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
            scores[task_id] = 0.0

    # Final summary
    print(f"\n[DEBUG] {'='*40}", flush=True)
    print(f"[DEBUG] BASELINE RESULTS", flush=True)
    print(f"[DEBUG] Task 1 (Easy):   {scores.get(1, 0.0):.4f}", flush=True)
    print(f"[DEBUG] Task 2 (Medium): {scores.get(2, 0.0):.4f}", flush=True)
    print(f"[DEBUG] Task 3 (Hard):   {scores.get(3, 0.0):.4f}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"[DEBUG] Average:         {avg:.4f}", flush=True)
    elapsed = (time.time() - START_TIME) / 60
    print(f"[DEBUG] Completed in {elapsed:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
