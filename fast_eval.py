import sys
import traceback

with open("test_err.log", "w") as f:
    try:
        from stable_baselines3 import PPO
        from env.gym_wrapper import ProcurementGymWrapper
        import numpy as np
        
        env = ProcurementGymWrapper(task=1, seed=42)
        model = PPO.load("models/task1_ppo")
        f.write("Task 1 initialized successfully.\n")
        
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        f.write(f"Task 1 action prediction: {action}\n")
    except Exception as e:
        f.write(traceback.format_exc())
