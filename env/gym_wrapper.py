import gymnasium as gym
import numpy as np
from gymnasium import spaces
from env.global_procurement_env import GlobalProcurementEnv


class ProcurementGymWrapper(gym.Env):
    """
    Gymnasium wrapper around GlobalProcurementEnv.
    This lets SB3's PPO train on our custom environment.

    Translates between:
      - GlobalProcurementEnv's Pydantic returns (Observation, Reward, bool, dict)
      - Gymnasium's expected API: np.ndarray obs, float reward, bool terminated/truncated
    """

    def __init__(self, task: int = 1, seed: int = 42):
        super().__init__()
        self.env = GlobalProcurementEnv()
        self.task = task
        self.seed = seed

        # Action space: discrete integers 0-6 (Task 3 uses all 7)
        # Task 1 uses 0-3, Task 2 uses 0-5, Task 3 uses 0-6
        n_actions = {1: 4, 2: 6, 3: 7}[task]
        self.action_space = spaces.Discrete(n_actions)

        # Observation space: a flat numpy array of the key state values
        # [budget_remaining, inventory_steel, inventory_chips, inventory_fabric,
        #  lead_days, carbon, violations, step_count]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(8,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        # Required by Gymnasium spec — initialises self.np_random so check_env passes
        super().reset(seed=seed)
        # env.reset() returns a Pydantic Observation — convert to numpy array
        obs_pydantic = self.env.reset(task=self.task, seed=self.seed)
        return self._obs_to_array(obs_pydantic), {}

    def step(self, action):
        # env.step() returns a 4-tuple: (Observation, Reward, bool, dict)
        observation, reward, done, info = self.env.step(int(action))
        obs_array = self._obs_to_array(observation)
        reward_float = float(reward.value)
        terminated = bool(done)
        truncated = False
        return obs_array, reward_float, terminated, truncated, {}

    def _obs_to_array(self, obs) -> np.ndarray:
        """
        Converts a Pydantic Observation (or any object with .model_dump()) into
        a normalised float32 array for the neural network.

        Fields sourced from Observation: budget_remaining, inventory, step,
          policy_violations_this_episode.
        Fields sourced from internal _state directly: lead_days, carbon.
        (These are tracked in SupplyChainState but not surfaced in Observation.)
        """
        MAX_BUDGET = 200000.0

        # Handle both Pydantic model instances and plain dicts gracefully
        if hasattr(obs, "model_dump"):
            obs_dict = obs.model_dump()
        else:
            obs_dict = obs

        # Read lead_days and carbon from internal state (not in Observation schema)
        internal = self.env._state
        lead_days = internal.lead_days if internal is not None else 0
        carbon = internal.carbon if internal is not None else 0.0

        inventory = obs_dict.get("inventory", {})

        return np.array([
            obs_dict.get("budget_remaining", 0.0) / MAX_BUDGET,
            inventory.get("steel", 0.0) / 1000.0,
            inventory.get("chips", 0.0) / 1000.0,
            inventory.get("fabric", 0.0) / 1000.0,
            lead_days / 30.0,
            carbon / 50.0,
            obs_dict.get("policy_violations_this_episode", 0) / 10.0,
            obs_dict.get("step", 0) / 100.0,
        ], dtype=np.float32)
