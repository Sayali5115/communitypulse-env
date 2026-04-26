"""
Training script for NGO Multi-Agent Coordination Environment
Trains agents using simple Q-learning approach
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
from ngo_coordination_env import NGOCoordinationEnv


class SimpleRLAgent:
    """
    Simple Q-learning based agent for demonstration
    In production, you'd use PPO, SAC, or MADDPG from Stable-Baselines3
    """
    
    def __init__(self, action_dim: int = 3, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.action_dim = action_dim
        
        # Initialize policy parameters (simple neural network weights)
        self.theta = np.random.randn(action_dim) * 0.1
        
        # Exploration parameters
        self.epsilon = 1.0  # Start with high exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
    
    def select_action(self, observation: Dict, explore: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy strategy"""
        # Extract state features
        urgency = observation['urgency'] / 10.0
        resources = observation['available_resources'][0] / 100.0
        people = observation['people_affected'][0] / 300.0
        
        # Simple feature vector
        state_features = np.array([urgency, resources, people])
        
        if explore and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.uniform(0, 1, size=self.action_dim)
        else:
            # Exploit: use learned policy
            # Simple linear policy: action = sigmoid(theta * features)
            raw_action = self.theta * np.mean(state_features)
            action = 1.0 / (1.0 + np.exp(-raw_action))  # Sigmoid
            action = np.clip(action, 0, 1)
        
        return action
    
    def update(self, observation: Dict, action: np.ndarray, reward: float, next_observation: Dict):
        """Update policy based on experience"""
        # Simple policy gradient update
        # In practice, use proper RL algorithms (PPO, SAC, etc.)
        
        # Gradient direction: increase probability of actions that gave high reward
        gradient = reward * action * 0.01
        self.theta += self.lr * np.mean(gradient)
        
        # Decay exploration
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def train_multi_agent_system(
    num_episodes: int = 100,
    max_steps: int = 50,
    num_agents: int = 3
) -> Tuple[List[float], List[Dict]]:
    """
    Train multiple agents in the NGO coordination environment
    
    Returns:
        rewards_history: List of total rewards per episode
        info_history: Detailed logs for analysis
    """
    # Create environment
    env = NGOCoordinationEnv(num_agents=num_agents, max_steps=max_steps)
    
    # Create agents
    agents = [SimpleRLAgent(action_dim=3) for _ in range(num_agents)]
    
    # Training history
    rewards_history = []
    info_history = []
    
    print("Starting Multi-Agent Training...")
    print(f"Episodes: {num_episodes}, Max Steps: {max_steps}, Agents: {num_agents}\n")
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        episode_info = {
            'episode': episode + 1,
            'steps': [],
            'task_type': info['task_type']
        }
        
        for step in range(max_steps):
            # Each agent selects action
            actions = np.vstack([agent.select_action(observation, explore=True) 
                                for agent in agents])
            
            # Environment step
            next_observation, reward, terminated, truncated, step_info = env.step(actions)
            
            # Update agents
            for i, agent in enumerate(agents):
                agent.update(observation, actions[i], reward, next_observation)
            
            # Track progress
            episode_reward += reward
            episode_info['steps'].append({
                'step': step + 1,
                'reward': reward,
                'allocations': step_info['allocations'].tolist()
            })
            
            observation = next_observation
            
            if terminated or truncated:
                break
        
        rewards_history.append(episode_reward)
        info_history.append(episode_info)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward (last 10): {avg_reward:.2f} | "
                  f"Task: {info['task_type']}")
    
    print("\nTraining Complete!")
    return rewards_history, info_history


def plot_learning_curves(rewards_history: List[float], info_history: List[Dict]):
    """Generate the 3 required visualizations"""
    
    # Calculate cumulative steps
    cumulative_steps = []
    total = 0
    for info in info_history:
        for step_info in info['steps']:
            total += 1
            cumulative_steps.append(total)
    
    # Extract all step rewards
    all_rewards = []
    for info in info_history:
        for step_info in info['steps']:
            all_rewards.append(step_info['reward'])
    
    # 1. OVERALL LEARNING CURVE
    plt.figure(figsize=(12, 6))
    plt.scatter(cumulative_steps, all_rewards, alpha=0.5, s=20, label='Step Rewards', color='green')
    
    # Smooth curve
    from scipy.ndimage import gaussian_filter1d
    if len(all_rewards) > 10:
        smoothed = gaussian_filter1d(all_rewards, sigma=5)
        plt.plot(cumulative_steps, smoothed, color='darkgreen', linewidth=2, label='Learning Curve')
    
    # Trend line
    z = np.polyfit(cumulative_steps, all_rewards, 1)
    p = np.poly1d(z)
    plt.plot(cumulative_steps, p(cumulative_steps), "--", color='blue', linewidth=2, label='Trend')
    
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Multi-Agent Learning Curve - Overall Progress', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/overall_learning_curve.png', dpi=150)
    print("Saved: results/overall_learning_curve.png")
    
    # 2. TASK COMPARISON (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    task_types = ['cooperation', 'competition', 'negotiation', 'coalition']
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, task_type in enumerate(task_types):
        # Filter episodes for this task
        task_episodes = [info for info in info_history if info['task_type'] == task_type]
        
        task_rewards = []
        task_steps = []
        step_counter = 0
        
        for ep_info in task_episodes:
            for step_info in ep_info['steps']:
                task_rewards.append(step_info['reward'])
                task_steps.append(step_counter)
                step_counter += 1
        
        if task_rewards:
            axes[idx].scatter(task_steps, task_rewards, alpha=0.6, s=15, color=colors[idx])
            
            if len(task_rewards) > 5:
                smoothed = gaussian_filter1d(task_rewards, sigma=3)
                axes[idx].plot(task_steps, smoothed, color=colors[idx], linewidth=2)
            
            axes[idx].set_title(f'Task: {task_type.capitalize()}', fontweight='bold')
            axes[idx].set_xlabel('Step')
            axes[idx].set_ylabel('Reward')
            axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/task_comparison.png', dpi=150)
    print("Saved: results/task_comparison.png")
    
    # 3. TASK PROGRESSION (bar chart)
    plt.figure(figsize=(10, 6))
    
    task_avg_rewards = []
    for task_type in task_types:
        task_rewards = []
        for info in info_history:
            if info['task_type'] == task_type:
                for step_info in info['steps']:
                    task_rewards.append(step_info['reward'])
        task_avg_rewards.append(np.mean(task_rewards) if task_rewards else 0)
    
    bars = plt.bar(range(len(task_types)), task_avg_rewards, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, task_avg_rewards)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', fontweight='bold')
    
    plt.xlabel('Task Type', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Task Progression - Average Rewards', fontsize=14, fontweight='bold')
    plt.xticks(range(len(task_types)), [t.capitalize() for t in task_types])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/task_progression.png', dpi=150)
    print("Saved: results/task_progression.png")
    
    plt.close('all')


def main():
    """Main execution"""
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Train agents
    rewards_history, info_history = train_multi_agent_system(
        num_episodes=100,  # 100 episodes total
        max_steps=50,      # 50 steps per episode
        num_agents=3       # 3 agents
    )
    
    # Generate visualizations
    print("\nGenerating Visualizations...")
    plot_learning_curves(rewards_history, info_history)
    
    # Save training data
    with open('results/training_results.json', 'w') as f:
        json.dump({
            'rewards_history': rewards_history,
            'info_history': info_history
        }, f, indent=2)
    print("Saved: results/training_results.json")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total Episodes: {len(rewards_history)}")
    print(f"First 10 Episodes Avg Reward: {np.mean(rewards_history[:10]):.2f}")
    print(f"Last 10 Episodes Avg Reward: {np.mean(rewards_history[-10:]):.2f}")
    improvement = ((np.mean(rewards_history[-10:]) - np.mean(rewards_history[:10])) / 
                   np.mean(rewards_history[:10]) * 100)
    print(f"Improvement: {improvement:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
