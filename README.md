# CommunityPulse-Env

An RL environment simulating humanitarian NGO volunteer coordination.

An AI agent acts as a field coordinator in an NGO control room — reading urgent humanitarian needs, allocating limited volunteers, and managing time pressure and resource scarcity. One decision per step. Dense rewards after every action.

---

## Motivation

Field coordinators at grassroots NGOs manually triage dozens of incoming needs, match volunteers by skill and availability, and race against deadlines — often with fewer resources than required. This environment models that exact decision process, enabling agents to learn optimal allocation strategies under real-world constraints.

No such training benchmark existed for this domain. CommunityPulse-Env fills that gap.

---

## Environment Description

The agent sits in a simulated NGO control room with:
- A set of **humanitarian needs** (medical, food, water, shelter, logistics) each with urgency, confidence, people affected, required skill, and optional deadline
- A set of **volunteers** with specific skills and availability
- A **step budget** that creates time pressure

Each step the agent takes one atomic action:
- **assign** — assign a volunteer to a need
- **wait** — do nothing this step
- **investigate** — spend a step to increase confidence on an uncertain need

The environment evolves after every action: needs get resolved, volunteers become busy, deadlines tick down, warnings appear.

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
| High urgency need resolved (urgency ≥ 0.8) | +0.5 × urgency |
| Deadline bonus | +0.5 |
| Impact bonus (people affected) | up to +0.3 |
| Wrong skill assigned | -0.5 |
| Busy volunteer assigned | -0.3 |
| Already resolved need | -0.2 |
| Lazy wait (volunteers available) | -0.3 |
| HIGH urgency need expired | -2.0 |
| Loop detection | -2.0 |

---

## Tasks

### Task 1 — Clean Allocation (Easy)
- 4 needs, 4 volunteers, 10 step budget
- Clear urgency signals, no deadlines, enough volunteers
- Tests basic allocation capability
- Grader: correct skill assignments / total needs

### Task 2 — Prioritization Under Scarcity (Medium)
- 6 needs, 3 volunteers, 14 step budget
- Fewer volunteers than needs, some urgency uncertainty
- Agent must prioritize HIGH urgency needs
- Grader: 60% urgency-weighted coverage + 40% skill match

### Task 3 — Deadline + Skill Optimization (Hard)
- 10 needs, 4 volunteers, 20 step budget
- Tight deadlines, skill constraints, maximum scarcity
- Agent must optimize coverage, skill quality, and deadline adherence
- Grader: 40% coverage + 35% skill match + 25% deadline adherence

---

## Baseline Scores

| Task | Score |
|---|---|
| Task 1 (Easy) | 1.0000 |
| Task 2 (Medium) | 0.9643 |
| Task 3 (Hard) | 0.9500 |
| Average | 0.9714 |

Model: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router

---

## Setup & Usage

### Local Setup

```bash
# Clone the repo
git clone https://github.com/Sayali5115/communitypulse-env
cd communitypulse-env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Run Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your-hf-token-here

python inference.py
```

### Docker

```bash
docker build -t communitypulse-env .

docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e HF_TOKEN=your-hf-token-here \
  communitypulse-env
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Take one action |
| `/state` | GET | Current observation |
| `/tasks` | GET | List all tasks |

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace API key | — |

---

## Project Structure

```
communitypulse-env/
├── inference.py          # Baseline inference script
├── config.py             # Central configuration
├── Dockerfile            # Container definition
├── openenv.yaml          # OpenEnv spec
├── requirements.txt
├── README.md
└── app/
    ├── main.py           # FastAPI endpoints
    ├── env.py            # Core state machine
    ├── models.py         # Pydantic models
    ├── graders.py        # Task graders
    └── data/
        ├── reports.json  # Synthetic needs data
        └── volunteers.json
```

---

## Real-World Connection

This environment was extracted from **CommunityPulse** — a multi-tenant SaaS platform being built for grassroots NGO volunteer coordination. The allocation decisions modeled here represent the exact decisions field coordinators make manually today. A trained agent could be deployed directly inside CommunityPulse to assist coordinators in real operations.
