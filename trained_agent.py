from stable_baselines3 import PPO
from env.gym_wrapper import ProcurementGymWrapper
import numpy as np


def evaluate_model(task_id: int, model_path: str, n_episodes: int = 5) -> float:
    model = PPO.load(model_path)
    env = ProcurementGymWrapper(task=task_id, seed=42)
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)

    avg = np.mean(rewards)
    print(f"Task {task_id}: avg_reward over {n_episodes} episodes = {avg:.4f}")
    return avg


if __name__ == "__main__":
    evaluate_model(1, "models/task1_ppo")
    # Task 2 and 3 added on Day 4
