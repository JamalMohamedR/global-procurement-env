from env.gym_wrapper import ProcurementGymWrapper
import gymnasium as gym

env = ProcurementGymWrapper(task=1, seed=42)

# Gymnasium's own checker validates your wrapper is correct
from gymnasium.utils.env_checker import check_env
check_env(env, warn=True)  # should print no errors

obs, info = env.reset()
print(obs.shape)   # should be (8,)
print(obs.dtype)   # should be float32
print(all(0.0 <= v <= 1.0 for v in obs))  # True — all values normalised

obs2, reward, term, trunc, info = env.step(0)
print(isinstance(reward, float))  # True
