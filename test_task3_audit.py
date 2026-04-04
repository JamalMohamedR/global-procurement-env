from stable_baselines3 import PPO
from env.gym_wrapper import ProcurementGymWrapper

model = PPO.load("models/task3_ppo")
env = ProcurementGymWrapper(task=3, seed=42)
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
print(0 <= int(action) <= 6)  # True — valid action range for Task 3
