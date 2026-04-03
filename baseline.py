import requests
import random

BASE_URL = "http://localhost:7860"  # Update to live HF URL on Day 4

def run_task(task_id: int, seed: int = 42) -> float:
    """Runs one episode with a random agent and returns the grader score."""
    # Reset the environment for this task
    reset_resp = requests.post(f"{BASE_URL}/reset", json={"task": task_id, "seed": seed})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    done = False
    episode_log = []

    # Determine valid action range per task
    max_action = {1: 3, 2: 5, 3: 6}[task_id]

    while not done:
        action = random.randint(0, max_action)
        step_resp = requests.post(f"{BASE_URL}/step", json={"action": action})
        step_resp.raise_for_status()
        result = step_resp.json()

        episode_log.append({
            "step": result.get("step"),
            "action": action,
            "reward": result.get("reward"),
            "done": result.get("done"),
        })
        done = result.get("done", False)

    # Get grader score via state endpoint
    state_resp = requests.get(f"{BASE_URL}/state")
    score = state_resp.json().get("grader_score", 0.0)
    return score or 0.0

if __name__ == "__main__":
    print("Running baseline random agent...")
    for task in [1, 2, 3]:
        score = run_task(task_id=task, seed=42)
        print(f"  Task {task}: baseline score = {score:.4f}")
