from stable_baselines3 import PPO
model = PPO.load("models/task2_ppo")
print("Model loaded successfully")
print(model.policy)
