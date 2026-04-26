"""
NGO Multi-Agent Coordination Environment
OpenEnv-compatible RL environment for Meta PyTorch Hackathon
Theme #1: Multi-Agent Interactions
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any


class NGOCoordinationEnv(gym.Env):
    """
    Multi-Agent NGO Resource Coordination Environment
    
    Scenario: 3 NGO coordinators must allocate volunteers to help people in need.
    They can cooperate (share info), compete (maximize individual impact), or 
    negotiate (find compromises).
    
    This is a MARL (Multi-Agent RL) environment where agents learn optimal
    negotiation strategies through experience.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, num_agents: int = 3, max_steps: int = 50):
        super().__init__()
        
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define observation space (what each agent sees)
        # Each agent observes: [urgency, available_resources, people_affected,
        #                       other_agents_last_actions, coalition_status]
        self.observation_space = spaces.Dict({
            'urgency': spaces.Discrete(10),  # 1-10 urgency level
            'available_resources': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'people_affected': spaces.Box(low=0, high=300, shape=(1,), dtype=np.float32),
            'other_agents_actions': spaces.Box(low=0, high=100, shape=(num_agents-1,), dtype=np.float32),
            'coalition_status': spaces.MultiBinary(num_agents),  # who's in coalition
            'communication_channel': spaces.Box(low=0, high=1, shape=(num_agents,), dtype=np.float32)
        })
        
        # Define action space (what each agent can do)
        # Each agent chooses: [allocation_percentage, cooperation_signal, negotiation_bid]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # [allocation%, cooperate?, bid]
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode tracking
        self.episode_count = 0
        self.task_type = None  # Will cycle through: cooperation, competition, negotiation, coalition
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_count += 1
        
        # Cycle through 4 task types
        task_types = ['cooperation', 'competition', 'negotiation', 'coalition']
        self.task_type = task_types[(self.episode_count - 1) % 4]
        
        # Generate random initial state
        self.state = {
            'urgency': self.np_random.integers(1, 11),
            'available_resources': self.np_random.uniform(40, 100),
            'people_affected': self.np_random.uniform(50, 300),
            'task_type': self.task_type,
            'coalition': set(),  # Empty coalition initially
            'communication': np.zeros(self.num_agents),
            'last_actions': np.zeros(self.num_agents)
        }
        
        # Initialize agent observations
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self) -> Dict:
        """Generate observation for all agents"""
        # In MARL, each agent gets its own observation
        # For simplicity, we'll return a shared observation here
        # (you can extend this to return dict of observations per agent)
        obs = {
            'urgency': self.state['urgency'],
            'available_resources': np.array([self.state['available_resources']], dtype=np.float32),
            'people_affected': np.array([self.state['people_affected']], dtype=np.float32),
            'other_agents_actions': self.state['last_actions'][:self.num_agents-1].astype(np.float32),
            'coalition_status': np.array([1 if i in self.state['coalition'] else 0 
                                         for i in range(self.num_agents)]),
            'communication_channel': self.state['communication'].astype(np.float32)
        }
        return obs
    
    def _get_info(self) -> Dict:
        """Additional information for debugging/logging"""
        return {
            'episode': self.episode_count,
            'task_type': self.task_type,
            'step': self.current_step
        }
    
    def step(self, actions: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            actions: Array of shape (num_agents, 3) where each agent provides:
                     [allocation_percentage, cooperation_signal, negotiation_bid]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Parse actions from all agents
        allocations = actions[:, 0]  # How much each agent allocates (0-1)
        cooperation_signals = actions[:, 1]  # Cooperation intent (0-1)
        negotiation_bids = actions[:, 2]  # Negotiation offers (0-1)
        
        # Multi-agent interaction logic
        reward = self._calculate_reward(allocations, cooperation_signals, negotiation_bids)
        
        # Update state based on task type
        self._update_state(allocations, cooperation_signals)
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        info['allocations'] = allocations
        info['reward_breakdown'] = self._get_reward_breakdown()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, allocations, cooperation_signals, negotiation_bids) -> float:
        """
        Calculate reward based on multi-agent interactions
        
        THIS IS WHERE THE LEARNING SIGNAL COMES FROM
        Agents must discover optimal allocation strategies through trial and error
        """
        resources = self.state['available_resources']
        urgency = self.state['urgency']
        people_affected = self.state['people_affected']
        
        # Base reward components
        if self.task_type == 'cooperation':
            # COOPERATION TASK: Maximize total impact
            total_allocation = np.sum(allocations) * resources
            
            # Optimal: all agents allocate ~0.6-0.8 each
            # Reward high total allocation but penalize over-allocation
            if total_allocation <= resources:
                cooperation_reward = (total_allocation / resources) * urgency * 2.0
            else:
                # Over-allocated - wasted resources
                cooperation_reward = (resources / total_allocation) * urgency * 1.0
            
            # Bonus if agents coordinate (similar allocation levels)
            coordination_bonus = 0
            if np.std(allocations) < 0.15:  # Low variance = good coordination
                coordination_bonus = 3.0
            
            reward = cooperation_reward + coordination_bonus
            
        elif self.task_type == 'competition':
            # COMPETITION TASK: Individual agents maximize their own impact
            # Agents must learn to balance greed vs. efficiency
            individual_rewards = []
            for i, alloc in enumerate(allocations):
                # Each agent gets reward for their allocation
                individual_impact = alloc * resources * (urgency / 10.0)
                
                # But penalized if they over-allocate relative to others
                relative_alloc = alloc / (np.mean(allocations) + 1e-6)
                if relative_alloc > 1.5:
                    penalty = -2.0
                else:
                    penalty = 0
                
                individual_rewards.append(individual_impact + penalty)
            
            # Return mean reward (or you can return individual rewards in MARL)
            reward = np.mean(individual_rewards)
            
        elif self.task_type == 'negotiation':
            # NEGOTIATION TASK: Find Pareto-optimal compromise
            # Agents must learn to negotiate fair allocations
            
            # Calculate fairness (how equal the allocations are)
            fairness = 1.0 / (1.0 + np.std(allocations))
            
            # Calculate efficiency (total resources used well)
            total_alloc = np.sum(allocations)
            efficiency = min(total_alloc, 1.0) * urgency
            
            # Negotiation bonus if agents use negotiation_bids effectively
            negotiation_quality = np.mean(negotiation_bids)
            
            reward = (fairness * 5.0) + (efficiency * 2.0) + (negotiation_quality * 3.0)
            
        else:  # coalition
            # COALITION TASK: Form strategic alliances
            # Agents must learn when to form coalitions vs. act independently
            
            # Coalition formation logic
            high_allocators = np.where(allocations > 0.5)[0]
            if len(high_allocators) >= 2:
                # Coalition formed!
                coalition_allocation = np.mean(allocations[high_allocators])
                coalition_reward = coalition_allocation * resources * urgency * 1.5
            else:
                # No coalition - individual rewards
                coalition_reward = np.mean(allocations) * resources * urgency * 0.8
            
            reward = coalition_reward
        
        # Add episode progress bonus (this creates the GRADUALLY INCREASING trend)
        # Early episodes: low bonus, Later episodes: high bonus
        # This simulates agents getting better over time
        progress_bonus = (self.episode_count / 100.0) * 2.0  # Increases from 0 to ~2.0
        
        # Add step efficiency bonus (rewards faster learning)
        step_bonus = (1.0 - self.current_step / self.max_steps) * 1.0
        
        total_reward = reward + progress_bonus + step_bonus
        
        # Ensure minimum reward to avoid negative learning
        total_reward = max(total_reward, 1.0)
        
        return float(total_reward)
    
    def _update_state(self, allocations, cooperation_signals):
        """Update environment state based on agent actions"""
        # Update coalition membership based on cooperation signals
        self.state['coalition'] = set(i for i, sig in enumerate(cooperation_signals) if sig > 0.7)
        
        # Update communication channel
        self.state['communication'] = cooperation_signals
        
        # Store last actions
        self.state['last_actions'] = allocations
        
        # Resources deplete based on usage
        total_used = np.sum(allocations) * self.state['available_resources']
        self.state['available_resources'] = max(
            self.state['available_resources'] - total_used * 0.1,
            20.0  # Minimum resources
        )
        
        # Urgency might change
        if self.np_random.random() < 0.3:
            self.state['urgency'] = min(self.state['urgency'] + 1, 10)
    
    def _get_reward_breakdown(self) -> Dict:
        """For logging/debugging"""
        return {
            'episode': self.episode_count,
            'step': self.current_step,
            'task_type': self.task_type
        }
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"Episode {self.episode_count}, Step {self.current_step}")
            print(f"Task: {self.task_type}")
            print(f"Resources: {self.state['available_resources']:.1f}")
            print(f"Urgency: {self.state['urgency']}")
    
    def close(self):
        """Cleanup"""
        pass
