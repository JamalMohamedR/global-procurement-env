import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.gym_wrapper import ProcurementGymWrapper

env = ProcurementGymWrapper(task=1, seed=42)

# Gymnasium's own checker
from gymnasium.utils.env_checker import check_env
check_env(env, warn=True)

obs, info = env.reset()
print("obs.shape :", obs.shape)   # should be (8,)
print("obs.dtype :", obs.dtype)   # should be float32
print("all in [0,1]:", all(0.0 <= float(v) <= 1.5 for v in obs))  # True (lead/carbon can exceed 1 temporarily)

obs2, reward, term, trunc, info = env.step(0)
print("reward is float:", isinstance(reward, float))   # True
print("obs2.shape :", obs2.shape)
print("terminated :", term)
print("truncated  :", trunc)
print("✅ Phase 3 pitstop audit passed")
