"""
Multi-Agent Coordinator Environment for CommunityPulse
Extends base environment to support multiple coordinators with negotiation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import random
import copy

@dataclass
class Coordinator:
    """Represents one AI coordinator in multi-agent scenario."""
    id: str
    name: str
    assigned_needs: List[str] = field(default_factory=list)
    volunteer_pool: List[str] = field(default_factory=list)
    score: float = 0.0
    negotiations_sent: int = 0
    negotiations_accepted: int = 0
    
@dataclass
class Negotiation:
    """Represents a resource negotiation between coordinators."""
    id: str
    from_coordinator: str
    to_coordinator: str
    offer_type: str  # "volunteer_swap", "volunteer_loan", "intel_share"
    volunteer_id: Optional[str] = None
    need_id: Optional[str] = None
    status: str = "pending"  # pending, accepted, rejected
    expiry_step: int = -1

class MultiAgentCoordinatorEnv:
    """
    Multi-agent extension of CommunityPulse environment.
    
    Features:
    - 2-3 coordinators operating simultaneously
    - Shared volunteer pool (30% overlap)
    - Negotiation actions: swap, loan, share intel
    - Competitive scoring with coalition benefits
    """
    
    def __init__(self, base_env, num_coordinators: int = 2):
        """
        Initialize multi-agent environment.
        
        Args:
            base_env: Instance of base CommunityPulseEnv
            num_coordinators: Number of coordinators (2-3)
        """
        self.base_env = base_env
        self.num_coordinators = min(max(num_coordinators, 2), 3)
        
        self.coordinators: Dict[str, Coordinator] = {}
        self.negotiations: Dict[str, Negotiation] = {}
        self.negotiation_counter = 0
        
        self.shared_volunteer_ratio = 0.3  # 30% volunteers are shared
        self.current_coordinator = "coord_1"  # Active coordinator
        self.step_count = 0
        
    def reset(self, task_id: int = 1) -> Dict[str, Any]:
        """Reset environment with multi-coordinator setup."""
        # Reset base environment
        obs = self.base_env.reset(task_id)
        
        # Convert Observation object to dict if needed
        if hasattr(obs, 'dict'):
            obs = obs.dict()
        elif hasattr(obs, 'model_dump'):
            obs = obs.model_dump()
        
        # Initialize coordinators
        self.coordinators = {}
        for i in range(self.num_coordinators):
            coord_id = f"coord_{i+1}"
            self.coordinators[coord_id] = Coordinator(
                id=coord_id,
                name=f"Coordinator {i+1}",
                volunteer_pool=[],
                assigned_needs=[]
            )
        
        # Distribute volunteers across coordinators with overlap
        all_volunteers = [v["id"] for v in obs["volunteers"]]
        
        # Each coordinator gets base allocation
        base_size = len(all_volunteers) // self.num_coordinators
        
        for i, coord_id in enumerate(self.coordinators.keys()):
            # Give base allocation
            start_idx = i * base_size
            end_idx = start_idx + base_size if i < self.num_coordinators - 1 else len(all_volunteers)
            base_volunteers = all_volunteers[start_idx:end_idx]
            
            # Add shared volunteers (30% from adjacent pool)
            if i < self.num_coordinators - 1:
                next_pool_size = min(base_size, len(all_volunteers) - end_idx)
                shared_count = int(next_pool_size * self.shared_volunteer_ratio)
                shared_volunteers = all_volunteers[end_idx:end_idx + shared_count]
                base_volunteers.extend(shared_volunteers)
            
            self.coordinators[coord_id].volunteer_pool = base_volunteers
        
        # Distribute needs
        all_needs = [n["id"] for n in obs["needs"]]
        random.shuffle(all_needs)
        
        needs_per_coord = len(all_needs) // self.num_coordinators
        for i, coord_id in enumerate(self.coordinators.keys()):
            start_idx = i * needs_per_coord
            end_idx = start_idx + needs_per_coord if i < self.num_coordinators - 1 else len(all_needs)
            self.coordinators[coord_id].assigned_needs = all_needs[start_idx:end_idx]
        
        self.negotiations = {}
        self.negotiation_counter = 0
        self.step_count = 0
        self.current_coordinator = "coord_1"
        
        return self._get_multiagent_observation(obs)
    
    def step(self, action: Dict[str, Any], coordinator_id: str = None) -> Dict[str, Any]:
        """
        Execute action in multi-agent environment.
        
        Args:
            action: Action dict with type, and params
            coordinator_id: Which coordinator is acting (defaults to current)
        
        Returns:
            Observation dict with multi-agent context
        """
        if coordinator_id is None:
            coordinator_id = self.current_coordinator
        
        reward = 0.0
        info = {"coordinator": coordinator_id}
        
        action_type = action.get("type")
        
        # Handle negotiation actions
        if action_type == "negotiate":
            reward, info = self._handle_negotiation(action, coordinator_id)
            
        # Handle standard actions (assign, wait, investigate)
        elif action_type in ["assign", "wait", "investigate"]:
            # Validate coordinator has access to resources
            if action_type == "assign":
                volunteer_id = action.get("volunteer_id")
                if volunteer_id not in self.coordinators[coordinator_id].volunteer_pool:
                    reward = -1.0
                    info["error"] = "volunteer_not_in_pool"
                else:
                    # Execute in base environment
                    obs_b, rew_b, done_b, info_b = self.base_env.step(action)
                    reward = rew_b.value
                    info.update(info_b or {})
                    
            else:
                    # Execute in base environment
                    obs_b, rew_b, done_b, info_b = self.base_env.step(action)
                    reward = rew_b.value
                    info.update(info_b or {})
        
        else:
            reward = -0.5
            info["error"] = f"unknown_action_type_{action_type}"
        
        # Update coordinator score
        self.coordinators[coordinator_id].score += reward
        
        # Process pending negotiations
        self._process_negotiations()
        
        # Rotate to next coordinator
        self._rotate_coordinator()
        
        self.step_count += 1
        
        # Get updated observation
        base_obs = self.base_env.state()
        
        # Convert to dict if needed
        if hasattr(base_obs, 'dict'):
            base_obs = base_obs.dict()
        elif hasattr(base_obs, 'model_dump'):
            base_obs = base_obs.model_dump()
        
        obs = self._get_multiagent_observation(base_obs)
        
        return {
            "observation": obs,
            "reward": reward,
            "done": base_obs.get("done", False),
            "info": info
        }
    
    def _handle_negotiation(self, action: Dict[str, Any], coordinator_id: str) -> tuple:
        """Handle negotiation actions between coordinators."""
        offer_type = action.get("offer_type")
        target_coord = action.get("target_coordinator")
        
        if target_coord not in self.coordinators:
            return -0.3, {"error": "invalid_target_coordinator"}
        
        if target_coord == coordinator_id:
            return -0.3, {"error": "cannot_negotiate_with_self"}
        
        # Create negotiation
        neg_id = f"neg_{self.negotiation_counter}"
        self.negotiation_counter += 1
        
        negotiation = Negotiation(
            id=neg_id,
            from_coordinator=coordinator_id,
            to_coordinator=target_coord,
            offer_type=offer_type,
            volunteer_id=action.get("volunteer_id"),
            need_id=action.get("need_id"),
            status="pending",
            expiry_step=self.step_count + 3  # Expires in 3 steps
        )
        
        self.negotiations[neg_id] = negotiation
        self.coordinators[coordinator_id].negotiations_sent += 1
        
        # Auto-evaluate negotiation (simulated intelligence)
        # In real training, target coordinator would decide
        acceptance_probability = self._evaluate_negotiation(negotiation)
        
        if random.random() < acceptance_probability:
            negotiation.status = "accepted"
            reward = self._execute_negotiation(negotiation)
            self.coordinators[coordinator_id].negotiations_accepted += 1
            return reward, {"negotiation_accepted": True, "negotiation_id": neg_id}
        else:
            negotiation.status = "rejected"
            return -0.1, {"negotiation_rejected": True, "negotiation_id": neg_id}
    
    def _evaluate_negotiation(self, negotiation: Negotiation) -> float:
        """Simulate target coordinator evaluating negotiation offer."""
        # Simple heuristic: accept if offer benefits both parties
        offer_type = negotiation.offer_type
        
        if offer_type == "volunteer_swap":
            # Accept if target coordinator is overloaded
            target = self.coordinators[negotiation.to_coordinator]
            if len(target.assigned_needs) > len(target.volunteer_pool):
                return 0.7
            return 0.3
        
        elif offer_type == "volunteer_loan":
            # Accept if volunteer is idle
            return 0.6
        
        elif offer_type == "intel_share":
            # Always valuable
            return 0.8
        
        return 0.5
    
    def _execute_negotiation(self, negotiation: Negotiation) -> float:
        """Execute accepted negotiation and return reward."""
        from_coord = self.coordinators[negotiation.from_coordinator]
        to_coord = self.coordinators[negotiation.to_coordinator]
        
        if negotiation.offer_type == "volunteer_swap":
            # Swap one volunteer between pools
            if negotiation.volunteer_id in from_coord.volunteer_pool:
                from_coord.volunteer_pool.remove(negotiation.volunteer_id)
                to_coord.volunteer_pool.append(negotiation.volunteer_id)
                return 1.0  # Successful coalition formation
        
        elif negotiation.offer_type == "volunteer_loan":
            # Temporary loan (returns after 3 steps)
            if negotiation.volunteer_id in from_coord.volunteer_pool:
                to_coord.volunteer_pool.append(negotiation.volunteer_id)
                return 0.5
        
        elif negotiation.offer_type == "intel_share":
            # Share information about need (increases confidence)
            return 0.3
        
        return 0.0
    
    def _process_negotiations(self):
        """Process and expire pending negotiations."""
        expired = []
        for neg_id, negotiation in self.negotiations.items():
            if negotiation.status == "pending" and self.step_count >= negotiation.expiry_step:
                negotiation.status = "expired"
                expired.append(neg_id)
        
        # Clean up expired
        for neg_id in expired:
            del self.negotiations[neg_id]
    
    def _rotate_coordinator(self):
        """Rotate to next coordinator for turn-based action."""
        coord_ids = list(self.coordinators.keys())
        current_idx = coord_ids.index(self.current_coordinator)
        next_idx = (current_idx + 1) % len(coord_ids)
        self.current_coordinator = coord_ids[next_idx]
    
    def _get_multiagent_observation(self, base_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Augment base observation with multi-agent context."""
        # Convert to dict if it's a Pydantic model
        if hasattr(base_obs, 'dict'):
            base_obs = base_obs.dict()
        elif hasattr(base_obs, 'model_dump'):
            base_obs = base_obs.model_dump()
        
        obs = copy.deepcopy(base_obs)
        
        # Add coordinator context
        obs["current_coordinator"] = self.current_coordinator
        obs["coordinators"] = []
        
        for coord_id, coord in self.coordinators.items():
            obs["coordinators"].append({
                "id": coord.id,
                "name": coord.name,
                "assigned_needs": coord.assigned_needs,
                "volunteer_pool_size": len(coord.volunteer_pool),
                "score": coord.score,
                "negotiations_sent": coord.negotiations_sent,
                "negotiations_accepted": coord.negotiations_accepted,
                "is_current": coord_id == self.current_coordinator
            })
        
        # Add active negotiations
        obs["negotiations"] = []
        for neg_id, negotiation in self.negotiations.items():
            if negotiation.status == "pending":
                obs["negotiations"].append({
                    "id": negotiation.id,
                    "from": negotiation.from_coordinator,
                    "to": negotiation.to_coordinator,
                    "type": negotiation.offer_type,
                    "expires_in": negotiation.expiry_step - self.step_count
                })
        
        # Filter volunteers to show only current coordinator's pool
        current_pool = self.coordinators[self.current_coordinator].volunteer_pool
        obs["volunteers"] = [
            v for v in obs["volunteers"]
            if v["id"] in current_pool
        ]
        
        # Filter needs to show only current coordinator's assignment
        current_needs = self.coordinators[self.current_coordinator].assigned_needs
        obs["needs"] = [
            n for n in obs["needs"]
            if n["id"] in current_needs
        ]
        
        return obs
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get coordinator rankings."""
        rankings = []
        for coord in self.coordinators.values():
            rankings.append({
                "id": coord.id,
                "name": coord.name,
                "score": coord.score,
                "needs_assigned": len(coord.assigned_needs),
                "negotiation_success_rate": (
                    coord.negotiations_accepted / coord.negotiations_sent
                    if coord.negotiations_sent > 0 else 0.0
                )
            })
        
        rankings.sort(key=lambda x: x["score"], reverse=True)
        return rankings
