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


BASE_URL = "https://JEN-chad-global-procurement-env.hf.space"
    
def evaluate_random(task_id: int, n_episodes: int = 5) -> float:
    env = ProcurementGymWrapper(task=task_id, seed=42)
    rewards = []
    # Determine valid action range per task
    max_action = {1: 3, 2: 5, 3: 6}[task_id]

    import random
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = random.randint(0, max_action)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)

    avg = np.mean(rewards)
    return avg

if __name__ == "__main__":
    baseline_scores = {}
    trained_scores = {}

    with open("results.txt", "w") as f:
        for task_id in [1, 2, 3]:
            # Get baseline
            base_avg = evaluate_random(task_id)
            baseline_scores[task_id] = base_avg
            
            # Get trained
            trained_avg = evaluate_model(task_id, f"models/task{task_id}_ppo")
            trained_scores[task_id] = trained_avg
            
            f.write(f"Task {task_id}: random_avg = {base_avg:.4f}, trained_avg = {trained_avg:.4f}\n")

        f.write("\n--- Final Comparison ---\n")
        print("\n--- Final Comparison ---")
        for t in [1, 2, 3]:
            res = f"Task {t}: random={baseline_scores[t]:.4f}, trained={trained_scores[t]:.4f}"
            print(res)
            f.write(res + "\n")
