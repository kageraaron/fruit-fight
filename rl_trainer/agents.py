from typing import Dict, List, Union
import random
from stable_baselines3 import PPO

class HeuristicBot:
    def get_action(self, state: Dict) -> str:
        raise NotImplementedError

class ConservativeBot(HeuristicBot):
    def get_action(self, state: Dict) -> str:
        # Bust rule: bust first.
        # Check if already busted in Engine logic? 
        # For heuristics, they should just mirror the game rules.
        if state["pendingSteal"]: return "STEAL"
        p_idx = state["activePlayerIndex"]
        player = state["players"][p_idx]
        if len(player["display"]) < 3: return "HIT"
        return "STAND"

class AggressiveBot(HeuristicBot):
    def get_action(self, state: Dict) -> str:
        if state["pendingSteal"]: return "STEAL"
        p_idx = state["activePlayerIndex"]
        player = state["players"][p_idx]
        if len(player["display"]) < 5: return "HIT"
        return "STAND"

class MathBot(HeuristicBot):
    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def get_action(self, state: Dict) -> str:
        # Precedence: Bust check first!
        # ... logic here ...
        # (Actually, heuristic bots are simple. If we want to fix their logic, 
        # we update their action evaluation to check bust first)
        
        # Simple bust risk check for bots
        p_idx = state["activePlayerIndex"]
        player = state["players"][p_idx]
        
        # Check for immediate bust if hit
        # ... (simplified)
        
        if state["pendingSteal"]: return "STEAL"
        
        if len(player["display"]) < 3: return "HIT"
        return "STAND"

class NeuralBot(HeuristicBot):
    def __init__(self, model: Union[PPO, str]):
        if isinstance(model, str):
            self.model = PPO.load(model)
        else:
            self.model = model

    def predict_action(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        # 0: HIT, 1: STAND, 2: STEAL, 3: SKIP_STEAL
        return action
