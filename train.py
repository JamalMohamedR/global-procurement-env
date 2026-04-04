from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.gym_wrapper import ProcurementGymWrapper


def train_task(task_id: int, total_timesteps: int, save_path: str):
    """Trains a PPO agent on a single task and saves the model."""
    print(f"Training Task {task_id} for {total_timesteps} timesteps...")

    # make_vec_env wraps the env in a vectorised container (SB3 requires this)
    env = make_vec_env(lambda: ProcurementGymWrapper(task=task_id, seed=42), n_envs=1)

    model = PPO(
        policy="MlpPolicy",  # Multi-layer perceptron — right choice for flat obs arrays
        env=env,
        verbose=1,           # Print training progress
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    env.close()


if __name__ == "__main__":
    # Day 3 already trained Task 1 — don't re-run that unless needed
    # train_task(task_id=1, total_timesteps=50000, save_path="models/task1_ppo")
    # train_task(task_id=2, total_timesteps=100000, save_path="models/task2_ppo")
    train_task(task_id=3, total_timesteps=50000, save_path="models/task3_ppo")
