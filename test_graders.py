from app.env import CommunityPulseEnv
from app.models import Action, ActionType
from app.graders import grade

# Test Task 1 - assign everything correctly
env = CommunityPulseEnv()
env.reset(1)
env.step(Action(type=ActionType.ASSIGN, need_id='n1', volunteer_id='v1'))
env.step(Action(type=ActionType.ASSIGN, need_id='n2', volunteer_id='v2'))
env.step(Action(type=ActionType.ASSIGN, need_id='n3', volunteer_id='v3'))
env.step(Action(type=ActionType.ASSIGN, need_id='n4', volunteer_id='v4'))
result = grade(env, 1)
print('Task 1 score:', result['score'])

# Test Task 2
env.reset(2)
env.step(Action(type=ActionType.ASSIGN, need_id='n1', volunteer_id='v1'))
env.step(Action(type=ActionType.ASSIGN, need_id='n2', volunteer_id='v3'))
env.step(Action(type=ActionType.ASSIGN, need_id='n5', volunteer_id='v5'))
result = grade(env, 2)
print('Task 2 score:', result['score'])

# Test Task 3
env.reset(3)
env.step(Action(type=ActionType.ASSIGN, need_id='n1', volunteer_id='v1'))
env.step(Action(type=ActionType.ASSIGN, need_id='n2', volunteer_id='v3'))
env.step(Action(type=ActionType.ASSIGN, need_id='n3', volunteer_id='v2'))
env.step(Action(type=ActionType.ASSIGN, need_id='n4', volunteer_id='v4'))
result = grade(env, 3)
print('Task 3 score:', result['score'])

print('graders.py OK')
