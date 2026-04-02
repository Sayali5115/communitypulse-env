"""
CommunityPulse-Env — Core Environment (State Machine)
Implements reset(), step(), state() per OpenEnv spec.
"""

import json
import uuid
import copy
from pathlib import Path
from typing import Tuple

from app.models import (
    Observation, Action, Reward,
    Need, Volunteer,
    NeedStatus, ActionType
)
from config import (
    TASK_BUDGETS,
    TASK_NEED_COUNTS,
    TASK_VOLUNTEER_COUNTS,
    REWARD_CORRECT_ASSIGN,
    REWARD_URGENCY_BONUS,
    REWARD_DEADLINE_BONUS,
    REWARD_INVESTIGATE_GAIN,
    PENALTY_WRONG_SKILL,
    PENALTY_UNAVAILABLE_VOL,
    PENALTY_ALREADY_RESOLVED,
    PENALTY_EXPIRED_NEED,
    PENALTY_LOOP,
    PENALTY_LAZY_WAIT,
)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

def _load_reports() -> dict:
    with open(DATA_DIR / "reports.json", "r") as f:
        return json.load(f)

def _load_volunteers() -> list:
    with open(DATA_DIR / "volunteers.json", "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# ENVIRONMENT CLASS
# ─────────────────────────────────────────────

class CommunityPulseEnv:
    """
    NGO Resource Allocation Environment.

    The agent allocates volunteers to humanitarian needs
    step by step, under time pressure and resource scarcity.

    State machine:
        reset(task_id) → initial Observation
        step(action)   → Observation, Reward, done, info
        state()        → current Observation (no step consumed)
    """

    def __init__(self):
        self._reports_data   = _load_reports()
        self._volunteers_data = _load_volunteers()

        # Episode state
        self.episode_id:     str            = ""
        self.task_id:        int            = 1
        self.step_count:     int            = 0
        self.budget:         int            = 0
        self.needs:          list[Need]     = []
        self.volunteers:     list[Volunteer]= []
        self.last_reward:    float          = 0.0
        self.last_action:    dict           = {}   # for loop detection
        self._done:          bool           = False

    # ─────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────

    def reset(self, task_id: int) -> Observation:
        """
        Start a new episode for the given task.
        Returns the initial observation.
        """
        self.episode_id  = str(uuid.uuid4())[:8]
        self.task_id     = task_id
        self.step_count  = 0
        self.budget      = TASK_BUDGETS[task_id]
        self.last_reward = 0.0
        self.last_action = {}
        self._done       = False

        # Load needs for this task
        task_key = f"task{task_id}"
        raw_needs = self._reports_data[task_key]["needs"]
        self.needs = [Need(**n) for n in raw_needs]

        # Load volunteers for this task
        vol_ids = self._reports_data[task_key]["volunteers"]
        all_vols = {v["id"]: v for v in self._volunteers_data}
        self.volunteers = [
            Volunteer(**{**all_vols[vid], "available": True, "busy_until_step": 0})
            for vid in vol_ids
        ]

        return self._build_observation()

    # ─────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        """
        Execute one action. Returns (observation, reward, done, info).
        """
        if self._done:
            obs = self._build_observation()
            reward = Reward(
                value=0.0,
                reason="Episode already finished. Call reset() to start a new episode.",
                done=True,
                info=self._episode_info()
            )
            return obs, reward, True, reward.info

        # Free up volunteers whose busy_until_step has passed
        self._release_volunteers()

        # Execute the action and get reward
        reward_value, reason = self._execute_action(action)

        # Consume one step
        self.step_count += 1
        self.last_reward = reward_value

        # Tick deadlines — expire needs that hit deadline
        self._tick_deadlines()

        # Check if episode is over
        done = self._check_done()
        self._done = done

        # End of episode penalty: unresolved HIGH urgency needs
        if done:
            penalty, penalty_reason = self._end_of_episode_penalty()
            reward_value += penalty
            if penalty < 0:
                reason += f" | End penalty: {penalty_reason}"

        self.last_reward = reward_value

        reward = Reward(
            value=round(reward_value, 3),
            reason=reason,
            done=done,
            info=self._episode_info()
        )

        obs = self._build_observation()
        return obs, reward, done, reward.info

    # ─────────────────────────────────────────
    # STATE
    # ─────────────────────────────────────────

    def state(self) -> Observation:
        """Return current observation without consuming a step."""
        return self._build_observation()

    # ─────────────────────────────────────────
    # ACTION EXECUTION
    # ─────────────────────────────────────────

    def _execute_action(self, action: Action) -> Tuple[float, str]:
        """
        Execute action and return (reward_value, reason_string).
        """

        # ── WAIT ──────────────────────────────
        if action.type == ActionType.WAIT:
            # Penalize waiting if volunteers are available and needs are open
            available_vols = [v for v in self.volunteers if v.available]
            open_needs     = [n for n in self.needs if n.status == NeedStatus.OPEN]

            if available_vols and open_needs:
                return PENALTY_LAZY_WAIT, "Waited with available volunteers and open needs."
            return 0.0, "Waited. No available volunteers or no open needs."

        # ── INVESTIGATE ───────────────────────
        if action.type == ActionType.INVESTIGATE:
            if not action.need_id:
                return -0.2, "Investigate action missing need_id."

            need = self._get_need(action.need_id)
            if need is None:
                return -0.2, f"Need {action.need_id} not found."
            if need.status != NeedStatus.OPEN:
                return -0.2, f"Need {action.need_id} is not open (status: {need.status})."

            # Loop detection
            last = self.last_action
            if last.get("type") == "investigate" and last.get("need_id") == action.need_id:
                self.last_action = {"type": "investigate", "need_id": action.need_id}
                return PENALTY_LOOP, f"Repeated investigate on {action.need_id}. Loop detected."

            # Boost confidence (capped at 1.0)
            old_conf = need.confidence
            need.confidence = min(1.0, need.confidence + 0.2)
            self.last_action = {"type": "investigate", "need_id": action.need_id}
            return REWARD_INVESTIGATE_GAIN, (
                f"Investigated {action.need_id}. "
                f"Confidence: {old_conf:.2f} → {need.confidence:.2f}"
            )

        # ── ASSIGN ────────────────────────────
        if action.type == ActionType.ASSIGN:
            if not action.need_id or not action.volunteer_id:
                return -0.2, "Assign action missing need_id or volunteer_id."

            need      = self._get_need(action.need_id)
            volunteer = self._get_volunteer(action.volunteer_id)

            # Validate need
            if need is None:
                return -0.2, f"Need {action.need_id} not found."
            if need.status == NeedStatus.RESOLVED:
                return PENALTY_ALREADY_RESOLVED, f"Need {action.need_id} already resolved."
            if need.status == NeedStatus.EXPIRED:
                return PENALTY_ALREADY_RESOLVED, f"Need {action.need_id} already expired."
            if need.status == NeedStatus.ASSIGNED:
                return PENALTY_ALREADY_RESOLVED, f"Need {action.need_id} already assigned."

            # Validate volunteer
            if volunteer is None:
                return -0.2, f"Volunteer {action.volunteer_id} not found."
            if not volunteer.available:
                return PENALTY_UNAVAILABLE_VOL, (
                    f"Volunteer {action.volunteer_id} is busy until step {volunteer.busy_until_step}."
                )

            # Loop detection
            last = self.last_action
            if (last.get("type") == "assign"
                    and last.get("need_id") == action.need_id
                    and last.get("volunteer_id") == action.volunteer_id):
                self.last_action = {
                    "type": "assign",
                    "need_id": action.need_id,
                    "volunteer_id": action.volunteer_id
                }
                return PENALTY_LOOP, "Repeated identical assign action. Loop detected."

            # Check skill match
            skill_match = (volunteer.skill == need.required_skill)

            if not skill_match:
                # Wrong skill — still assign but penalize
                need.status             = NeedStatus.ASSIGNED
                need.assigned_volunteer = volunteer.id
                volunteer.available     = False
                volunteer.busy_until_step = self.step_count + 3

                self.last_action = {
                    "type": "assign",
                    "need_id": action.need_id,
                    "volunteer_id": action.volunteer_id
                }
                return PENALTY_WRONG_SKILL, (
                    f"Assigned {volunteer.id} ({volunteer.skill}) to {need.id} "
                    f"which needs {need.required_skill}. Skill mismatch."
                )

            # ✅ Correct assignment
            need.status             = NeedStatus.RESOLVED
            need.assigned_volunteer = volunteer.id
            volunteer.available     = False
            volunteer.busy_until_step = self.step_count + 3

            reward = REWARD_CORRECT_ASSIGN

            # Urgency bonus
            urgency_bonus = 0.0
            if need.urgency >= 0.8:
                urgency_bonus = REWARD_URGENCY_BONUS * need.urgency
                reward += urgency_bonus

            # Deadline bonus
            deadline_bonus = 0.0
            if need.deadline_steps > 0:
                deadline_bonus = REWARD_DEADLINE_BONUS
                reward += deadline_bonus

            # Scale by people affected (normalized, max 500 people = 1.0 bonus)
            impact_bonus = round(min(need.people_affected / 500, 1.0) * 0.3, 3)
            reward += impact_bonus

            self.last_action = {
                "type": "assign",
                "need_id": action.need_id,
                "volunteer_id": action.volunteer_id
            }

            reason = (
                f"Correctly assigned {volunteer.id} ({volunteer.skill}) "
                f"to {need.id} (urgency={need.urgency}, "
                f"people={need.people_affected})."
            )
            if urgency_bonus > 0:
                reason += f" Urgency bonus: +{urgency_bonus:.2f}."
            if deadline_bonus > 0:
                reason += f" Deadline bonus: +{deadline_bonus:.2f}."

            return round(reward, 3), reason

        return 0.0, "Unknown action type."

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _get_need(self, need_id: str):
        for n in self.needs:
            if n.id == need_id:
                return n
        return None

    def _get_volunteer(self, vol_id: str):
        for v in self.volunteers:
            if v.id == vol_id:
                return v
        return None

    def _release_volunteers(self):
        """Free volunteers whose busy period has ended."""
        for v in self.volunteers:
            if not v.available and v.busy_until_step <= self.step_count:
                v.available = True

    def _tick_deadlines(self):
        """Decrement deadline counters. Expire needs that hit 0."""
        for need in self.needs:
            if need.status == NeedStatus.OPEN and need.deadline_steps > 0:
                need.deadline_steps -= 1
                if need.deadline_steps == 0:
                    need.status = NeedStatus.EXPIRED

    def _check_done(self) -> bool:
        """Episode ends when budget is exhausted or all needs are resolved/expired."""
        if self.step_count >= self.budget:
            return True
        active = [
            n for n in self.needs
            if n.status in (NeedStatus.OPEN, NeedStatus.ASSIGNED)
        ]
        return len(active) == 0

    def _end_of_episode_penalty(self) -> Tuple[float, str]:
        """Penalty for HIGH urgency needs left unresolved at episode end."""
        penalty = 0.0
        unresolved = []
        for need in self.needs:
            if need.status in (NeedStatus.OPEN, NeedStatus.ASSIGNED):
                if need.urgency >= 0.8:
                    penalty += PENALTY_EXPIRED_NEED
                    unresolved.append(need.id)
        if unresolved:
            return penalty, f"HIGH urgency needs unresolved: {unresolved}"
        return 0.0, ""

    def _build_observation(self) -> Observation:
        """Build the current Observation object."""
        warnings = self._build_warnings()
        return Observation(
            episode_id     = self.episode_id,
            task_id        = self.task_id,
            step           = self.step_count,
            time_remaining = max(0, self.budget - self.step_count),
            needs          = copy.deepcopy(self.needs),
            volunteers     = copy.deepcopy(self.volunteers),
            last_reward    = self.last_reward,
            warnings       = warnings
        )

    def _build_warnings(self) -> list[str]:
        """Generate warnings for the agent about urgent situations."""
        warnings = []

        # Warn about expiring deadlines
        for need in self.needs:
            if need.status == NeedStatus.OPEN and 0 < need.deadline_steps <= 3:
                warnings.append(
                    f"URGENT: Need {need.id} ({need.category}) "
                    f"expires in {need.deadline_steps} steps!"
                )

        # Warn about high urgency unassigned needs
        high_open = [
            n for n in self.needs
            if n.status == NeedStatus.OPEN and n.urgency >= 0.8
        ]
        if high_open:
            ids = [n.id for n in high_open]
            warnings.append(f"HIGH urgency needs still open: {ids}")

        # Warn about low budget
        remaining = self.budget - self.step_count
        if remaining <= 3:
            warnings.append(f"Only {remaining} steps remaining!")

        return warnings

    def _episode_info(self) -> dict:
        """Stats dict for logging and debugging."""
        resolved  = [n for n in self.needs if n.status == NeedStatus.RESOLVED]
        expired   = [n for n in self.needs if n.status == NeedStatus.EXPIRED]
        open_needs = [n for n in self.needs if n.status == NeedStatus.OPEN]
        assigned  = [n for n in self.needs if n.status == NeedStatus.ASSIGNED]
        busy_vols = [v for v in self.volunteers if not v.available]

        return {
            "step":              self.step_count,
            "budget":            self.budget,
            "time_remaining":    max(0, self.budget - self.step_count),
            "needs_total":       len(self.needs),
            "needs_resolved":    len(resolved),
            "needs_expired":     len(expired),
            "needs_open":        len(open_needs),
            "needs_assigned":    len(assigned),
            "volunteers_total":  len(self.volunteers),
            "volunteers_busy":   len(busy_vols),
        }
