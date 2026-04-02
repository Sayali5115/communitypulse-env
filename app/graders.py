"""
CommunityPulse-Env — Task Graders
Three deterministic graders, one per task.
All return float clamped to [0.0, 1.0].
"""

from app.models import Need, Volunteer, NeedStatus
from app.env import CommunityPulseEnv


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _clamp(value: float) -> float:
    """Hard clamp to [0.0, 1.0]. Always applied before returning."""
    return round(max(0.0, min(1.0, value)), 4)


# ─────────────────────────────────────────────
# GRADER 1 — BASIC ALLOCATION (EASY)
# ─────────────────────────────────────────────

def grade_task1(env: CommunityPulseEnv) -> dict:
    """
    Task 1 Grader: Basic Allocation

    Measures: Did the agent assign the right volunteer to the right need?

    Formula:
        score = correct_assignments / total_needs

    Correct assignment = volunteer skill matches need required_skill
                       + volunteer was available (not double-assigned)

    Returns dict with score and breakdown.
    """
    needs      = env.needs
    volunteers = env.volunteers

    total      = len(needs)
    correct    = 0
    breakdown  = []

    for need in needs:
        if need.status == NeedStatus.RESOLVED and need.assigned_volunteer:
            # Find the volunteer who was assigned
            vol = next(
                (v for v in volunteers if v.id == need.assigned_volunteer),
                None
            )
            if vol and vol.skill == need.required_skill:
                correct += 1
                breakdown.append({
                    "need_id":      need.id,
                    "result":       "correct",
                    "required":     need.required_skill,
                    "assigned":     vol.skill,
                })
            else:
                breakdown.append({
                    "need_id":      need.id,
                    "result":       "wrong_skill",
                    "required":     need.required_skill,
                    "assigned":     vol.skill if vol else "unknown",
                })
        else:
            breakdown.append({
                "need_id": need.id,
                "result":  "unresolved",
                "status":  need.status,
            })

    score = correct / total if total > 0 else 0.0

    return {
        "task":      1,
        "score":     _clamp(score),
        "correct":   correct,
        "total":     total,
        "breakdown": breakdown,
    }


# ─────────────────────────────────────────────
# GRADER 2 — PRIORITIZATION UNDER SCARCITY (MEDIUM)
# ─────────────────────────────────────────────

def grade_task2(env: CommunityPulseEnv) -> dict:
    """
    Task 2 Grader: Prioritization Under Scarcity

    Measures: Did the agent prioritize HIGH urgency needs
              when volunteers were limited?

    Formula:
        urgency_score = sum(urgency × people_affected for RESOLVED needs)
                      / sum(urgency × people_affected for ALL needs)

        skill_score   = correct_skill_assignments / total_assignments

        final = (0.6 × urgency_score) + (0.4 × skill_score)

    Returns dict with score and breakdown.
    """
    needs      = env.needs
    volunteers = env.volunteers

    # ── Urgency Score ──────────────────────────
    total_impact    = sum(n.urgency * n.people_affected for n in needs)
    resolved_impact = sum(
        n.urgency * n.people_affected
        for n in needs
        if n.status == NeedStatus.RESOLVED
    )
    urgency_score = resolved_impact / total_impact if total_impact > 0 else 0.0

    # ── Skill Score ────────────────────────────
    assignments = [n for n in needs if n.assigned_volunteer is not None]
    correct_skills = 0
    skill_breakdown = []

    for need in assignments:
        vol = next(
            (v for v in volunteers if v.id == need.assigned_volunteer),
            None
        )
        if vol and vol.skill == need.required_skill:
            correct_skills += 1
            skill_breakdown.append({
                "need_id":  need.id,
                "result":   "correct",
                "required": need.required_skill,
                "assigned": vol.skill,
            })
        else:
            skill_breakdown.append({
                "need_id":  need.id,
                "result":   "wrong_skill",
                "required": need.required_skill,
                "assigned": vol.skill if vol else "unknown",
            })

    skill_score = (
        correct_skills / len(assignments)
        if assignments else 0.0
    )

    # ── Final Score ────────────────────────────
    final = (0.6 * urgency_score) + (0.4 * skill_score)

    return {
        "task":           2,
        "score":          _clamp(final),
        "urgency_score":  round(urgency_score, 4),
        "skill_score":    round(skill_score, 4),
        "resolved_impact": round(resolved_impact, 2),
        "total_impact":   round(total_impact, 2),
        "skill_breakdown": skill_breakdown,
    }


# ─────────────────────────────────────────────
# GRADER 3 — DEADLINE + SKILL OPTIMIZATION (HARD)
# ─────────────────────────────────────────────

def grade_task3(env: CommunityPulseEnv) -> dict:
    """
    Task 3 Grader: Deadline + Skill Optimization

    Measures three things:
        1. Coverage:  fraction of HIGH urgency needs resolved
        2. Skill:     average skill match quality across all assignments
        3. Deadline:  fraction of deadline needs resolved before expiry

    Formula:
        coverage_score = HIGH urgency resolved / total HIGH urgency needs
        skill_score    = avg skill match (1.0 perfect, 0.5 partial, 0.0 wrong)
        deadline_score = needs_with_deadline resolved / total needs_with_deadline

        final = (0.4 × coverage) + (0.35 × skill) + (0.25 × deadline)

    Returns dict with score and breakdown.
    """
    needs      = env.needs
    volunteers = env.volunteers

    # ── Coverage Score ─────────────────────────
    high_needs     = [n for n in needs if n.urgency >= 0.8]
    high_resolved  = [n for n in high_needs if n.status == NeedStatus.RESOLVED]
    coverage_score = (
        len(high_resolved) / len(high_needs)
        if high_needs else 1.0
    )

    # ── Skill Score ────────────────────────────
    assigned_needs = [n for n in needs if n.assigned_volunteer is not None]
    skill_scores   = []

    for need in assigned_needs:
        vol = next(
            (v for v in volunteers if v.id == need.assigned_volunteer),
            None
        )
        if vol is None:
            skill_scores.append(0.0)
            continue

        if vol.skill == need.required_skill:
            skill_scores.append(1.0)       # perfect match
        elif vol.skill in ("medical", "logistics"):
            skill_scores.append(0.5)       # partial — generalist skills
        else:
            skill_scores.append(0.0)       # wrong skill

    skill_score = (
        sum(skill_scores) / len(skill_scores)
        if skill_scores else 0.0
    )

    # ── Deadline Score ─────────────────────────
    # Needs that originally had a deadline (we track by checking
    # if they were resolved vs expired)
    deadline_needs   = [
        n for n in needs
        if n.status in (NeedStatus.RESOLVED, NeedStatus.EXPIRED)
        and n.deadline_steps >= 0   # had a deadline
    ]
    deadline_resolved = [
        n for n in deadline_needs
        if n.status == NeedStatus.RESOLVED
    ]
    deadline_score = (
        len(deadline_resolved) / len(deadline_needs)
        if deadline_needs else 1.0
    )

    # ── Final Score ────────────────────────────
    final = (
        (0.40 * coverage_score) +
        (0.35 * skill_score)    +
        (0.25 * deadline_score)
    )

    return {
        "task":             3,
        "score":            _clamp(final),
        "coverage_score":   round(coverage_score, 4),
        "skill_score":      round(skill_score, 4),
        "deadline_score":   round(deadline_score, 4),
        "high_needs_total": len(high_needs),
        "high_resolved":    len(high_resolved),
        "deadline_needs":   len(deadline_needs),
        "deadline_resolved":len(deadline_resolved),
        "skill_details":    list(zip(
            [n.id for n in assigned_needs],
            skill_scores
        )),
    }


# ─────────────────────────────────────────────
# UNIFIED GRADER ENTRY POINT
# ─────────────────────────────────────────────

def grade(env: CommunityPulseEnv, task_id: int) -> dict:
    """
    Call the correct grader for the given task_id.
    Always returns a dict with at minimum:
        { "task": int, "score": float (0.0-1.0) }
    """
    if task_id == 1:
        return grade_task1(env)
    elif task_id == 2:
        return grade_task2(env)
    elif task_id == 3:
        return grade_task3(env)
    else:
        raise ValueError(f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")
