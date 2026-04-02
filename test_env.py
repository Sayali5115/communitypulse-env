from app.env import CommunityPulseEnv
from app.models import Action, ActionType

env = CommunityPulseEnv()

# ─────────────────────────────────────────────
# TEST 1: reset works
# ─────────────────────────────────────────────
obs = env.reset(1)
print("=== TASK 1 RESET ===")
print("Episode ID:", obs.episode_id)
print("Needs:", len(obs.needs))
print("Volunteers:", len(obs.volunteers))
print("Budget:", obs.time_remaining)

# ─────────────────────────────────────────────
# TEST 2: correct assign gives positive reward
# ─────────────────────────────────────────────
print("\n=== TASK 1 CORRECT ASSIGN ===")
action = Action(type=ActionType.ASSIGN, need_id="n1", volunteer_id="v1")
obs, reward, done, info = env.step(action)
print("Reward:", reward.value)
print("Reason:", reward.reason)
print("Done:", done)

# ─────────────────────────────────────────────
# TEST 3: wrong skill gives penalty
# ─────────────────────────────────────────────
print("\n=== TASK 1 WRONG SKILL ===")
action = Action(type=ActionType.ASSIGN, need_id="n2", volunteer_id="v3")
obs, reward, done, info = env.step(action)
print("Reward:", reward.value)
print("Reason:", reward.reason)

# ─────────────────────────────────────────────
# TEST 4: wait penalty when volunteers available
# ─────────────────────────────────────────────
print("\n=== TASK 1 LAZY WAIT ===")
action = Action(type=ActionType.WAIT)
obs, reward, done, info = env.step(action)
print("Reward:", reward.value)
print("Reason:", reward.reason)

# ─────────────────────────────────────────────
# TEST 5: investigate increases confidence
# ─────────────────────────────────────────────
print("\n=== TASK 2 INVESTIGATE ===")
obs = env.reset(2)
old_conf = obs.needs[0].confidence
action = Action(type=ActionType.INVESTIGATE, need_id="n1")
obs, reward, done, info = env.step(action)
new_conf = obs.needs[0].confidence
print("Reward:", reward.value)
print("Confidence:", old_conf, "->", new_conf)

# ─────────────────────────────────────────────
# TEST 6: full task 1 episode
# ─────────────────────────────────────────────
print("\n=== FULL TASK 1 EPISODE ===")
obs = env.reset(1)
actions = [
    Action(type=ActionType.ASSIGN, need_id="n1", volunteer_id="v1"),
    Action(type=ActionType.ASSIGN, need_id="n2", volunteer_id="v2"),
    Action(type=ActionType.ASSIGN, need_id="n3", volunteer_id="v3"),
    Action(type=ActionType.ASSIGN, need_id="n4", volunteer_id="v4"),
]
total_reward = 0
for a in actions:
    obs, reward, done, info = env.step(a)
    total_reward += reward.value
    print(f"  Step reward: {reward.value} | Done: {done}")

print("Total reward:", round(total_reward, 3))
print("Needs resolved:", info["needs_resolved"])

# ─────────────────────────────────────────────
# TEST 7: state() works without consuming step
# ─────────────────────────────────────────────
print("\n=== STATE() TEST ===")
obs = env.reset(1)
state1 = env.state()
state2 = env.state()
print("Step after two state() calls:", state2.step)
print("Should be 0 (state does not consume steps):", state2.step == 0)

print("\nenv.py OK")
