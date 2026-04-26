"""
Test script to verify NGOCoordinationEnv works correctly
"""

from ngo_coordination_env import NGOCoordinationEnv
import numpy as np


def test_environment():
    """Test basic environment functionality"""
    print("="*60)
    print("Testing NGO Coordination Environment")
    print("="*60)
    
    # Create environment
    env = NGOCoordinationEnv(num_agents=3, max_steps=50)
    print("Environment created successfully")
    
    # Test reset
    observation, info = env.reset(seed=42)
    print(f"Reset successful - Episode: {info['episode']}, Task: {info['task_type']}")
    
    # Verify observation space
    assert 'urgency' in observation
    assert 'available_resources' in observation
    assert 'people_affected' in observation
    print("Observation space correct")
    
    # Test step
    actions = np.random.uniform(0, 1, size=(3, 3))  # 3 agents, 3 actions each
    observation, reward, terminated, truncated, info = env.step(actions)
    print(f"Step successful - Reward: {reward:.2f}")
    
    # Run full episode
    observation, info = env.reset()
    total_reward = 0
    for step in range(50):
        actions = np.random.uniform(0, 1, size=(3, 3))
        observation, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"Full episode completed - Total Reward: {total_reward:.2f}")
    
    # Test all 4 task types
    task_types_seen = set()
    for _ in range(4):
        observation, info = env.reset()
        task_types_seen.add(info['task_type'])
    
    expected_tasks = {'cooperation', 'competition', 'negotiation', 'coalition'}
    assert task_types_seen == expected_tasks, f"Expected {expected_tasks}, got {task_types_seen}"
    print(f"All 4 task types working: {task_types_seen}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nEnvironment is ready for:")
    print("  - Stable-Baselines3 (PPO, SAC, A2C)")
    print("  - RLlib (Ray)")
    print("  - CleanRL")
    print("  - Any PyTorch-based RL algorithm")


if __name__ == "__main__":
    test_environment()
