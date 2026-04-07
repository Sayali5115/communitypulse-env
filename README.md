# CommunityPulse-Env

An RL environment simulating humanitarian NGO volunteer coordination.

An AI agent acts as a field coordinator — reading urgent humanitarian needs, allocating limited volunteers, and managing time pressure and resource scarcity. One decision per step. Dense rewards after every action.

---

## Motivation

Field coordinators at grassroots NGOs face a hard real-world problem every day: multiple urgent humanitarian needs arrive simultaneously, volunteers have different skills and availability, and every delay has a human cost. Today this coordination happens manually — a coordinator reads reports, makes judgment calls, and hopes they got the priorities right.

This environment models that exact decision process. An agent that learns here learns to prioritize under uncertainty, match skills to needs, and manage competing deadlines — skills that translate directly to real coordination scenarios.

No such training benchmark existed for this domain before.

---

## Environment Workflow

```
reset(task_id)
      │
      ▼
 Observation
 (needs + volunteers + warnings)
      │
      ▼
 Agent decides action          ┌─────────────────┐
 (assign / wait / investigate) │  action types:  │
      │                        │  assign         │
      ▼                        │  wait           │
 step(action) ──► Reward       │  investigate    │
      │           (dense)      └─────────────────┘
      ▼
 Episode done? ──No──► back to Observation
      │
     Yes
      ▼
 Grader scores (0.0 → 1.0)
```

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Git

### Step 1 — Clone the repo
```bash
git clone https://github.com/Sayali5115/communitypulse-env
cd communitypulse-env
```

### Step 2 — Create virtual environment
```bash
# Mac/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Start the environment server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Server is now running at `http://localhost:7860`

### Step 5 — Test the server (new terminal)
```bash
# Health check
curl http://localhost:7860/health
# Expected: {"status":"ok"}

# Reset to task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}'

# Take one action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"type": "assign", "need_id": "n1", "volunteer_id": "v1"}}'
```

### Step 6 — Run inference (new terminal)
```bash
# Set environment variables
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your-hf-token-here

# Windows
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
set HF_TOKEN=your-hf-token-here

# Run
python inference.py
```

Expected output:
```
[START] task=clean_allocation env=communitypulse-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"type":"assign","need_id":"n1","volunteer_id":"v1"} reward=1.48 done=false error=null
...
[END] success=true steps=9 score=1.000 rewards=1.48,1.52,...
```

---

## Docker

### Build and run
```bash
docker build -t communitypulse-env .

docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e HF_TOKEN=your-hf-token-here \
  communitypulse-env
```

### Test the container
```bash
curl http://localhost:7860/health
```

---

## API Endpoints

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/health` | GET | — | Health check |
| `/reset` | POST | `{"task_id": 1}` | Start new episode |
| `/step` | POST | `{"action": {...}}` | Take one action |
| `/state` | GET | — | Current observation |
| `/tasks` | GET | — | List all tasks |

---

## Action Space

```json
// Assign a volunteer to a need
{"type": "assign", "need_id": "n1", "volunteer_id": "v1"}

// Wait this step
{"type": "wait"}

// Investigate a need to increase confidence
{"type": "investigate", "need_id": "n2"}
```

---

## Observation Space

```json
{
  "episode_id": "abc123",
  "task_id": 1,
  "step": 3,
  "time_remaining": 7,
  "needs": [
    {
      "id": "n1",
      "category": "medical",
      "urgency": 0.9,
      "confidence": 1.0,
      "people_affected": 45,
      "deadline_steps": -1,
      "required_skill": "medical",
      "status": "open",
      "assigned_volunteer": null,
      "description": "Elderly patients need insulin at Dharavi camp."
    }
  ],
  "volunteers": [
    {
      "id": "v1",
      "name": "Priya Sharma",
      "skill": "medical",
      "available": true,
      "busy_until_step": 0
    }
  ],
  "last_reward": 1.477,
  "warnings": ["HIGH urgency needs still open: ['n1']"]
}
```

---

## Reward Function

| Event | Reward |
|---|---|
| Correct skill assignment | +1.0 |
| High urgency resolved (urgency ≥ 0.8) | +0.5 × urgency |
| Deadline bonus | +0.5 |
| Impact bonus (people affected) | up to +0.3 |
| Wrong skill assigned | -0.5 |
| Busy volunteer assigned | -0.3 |
| Lazy wait (volunteers available) | -0.3 |
| HIGH urgency need expired | -2.0 |
| Loop detection | -2.0 |

---

## Tasks

### Task 1 — Clean Allocation (Easy)
- 4 needs, 4 volunteers, 10 step budget
- Clear urgency, no deadlines, enough volunteers
- Tests basic skill-matching and allocation
- Grader: `correct_assignments / total_needs`

### Task 2 — Prioritization Under Scarcity (Medium)
- 6 needs, 3 volunteers, 14 step budget
- Fewer volunteers than needs, some urgency uncertainty
- Agent must prioritize HIGH urgency needs
- Grader: `(0.6 × urgency_weighted_coverage) + (0.4 × skill_match)`

### Task 3 — Deadline + Skill Optimization (Hard)
- 10 needs, 4 volunteers, 20 step budget
- Tight deadlines, skill constraints, maximum scarcity
- Agent must optimize coverage, skill quality, and deadline adherence
- Grader: `(0.4 × coverage) + (0.35 × skill_match) + (0.25 × deadline_adherence)`

---

## Baseline Scores

| Task | Score |
|---|---|
| Task 1 (Easy) | 1.0000 |
| Task 2 (Medium) | 0.9643 |
| Task 3 (Hard) | 0.9500 |
| Average | 0.9714 |

Model: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router
Runtime: ~1.8 minutes

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace API key | — |
| `ENV_URL` | Environment server URL | `http://localhost:7860` |

---

## Project Structure

```
communitypulse-env/
├── inference.py          # Baseline inference script (root)
├── config.py             # Central configuration
├── Dockerfile
├── openenv.yaml
├── requirements.txt
├── README.md
└── app/
    ├── main.py           # FastAPI endpoints
    ├── env.py            # Core state machine
    ├── models.py         # Pydantic models
    ├── graders.py        # Task graders
    └── data/
        ├── reports.json
        └── volunteers.json
```
