# Jenish (P1) — Day 3 Instructions
**Role:** Env Core | **Theme:** Baseline agent, Gymnasium wrapper, and first PPO training run

> ⚠️ **Today's Critical Gate:** Your trained PPO model for Task 1 must score higher than the random baseline. Models must be under 100MB.

---

## 🔗 Cross-Team Dependency — What You Need Today

**From Jamal (morning):** You need the FastAPI server running locally on `localhost:7860` (or `localhost:8000`) so that `baseline.py` can make real HTTP calls against it. Coordinate with Jamal at the start of the day — once he confirms the server is up, you can write `baseline.py` to point at it.

**From Jeswin (afternoon):** Jeswin should have run the full integration test (all 3 tasks via FastAPI) and confirmed grader scores vary. This confirms the task configs and graders are stable enough for you to train against.

---

## Phase 1 — Write `inference.py` (LLM Agent — Mandatory Submission File)

**What you're building:** The hackathon requires a file named exactly `inference.py` in the project root. This is the LLM-driven agent — it uses the OpenAI API client to have a language model choose actions, rather than picking randomly. **If this file is missing from the repo, the submission is automatically disqualified.**

**Three non-negotiable rules for `inference.py`:**
1. Must use `openai.OpenAI` client — not `requests`, not `httpx`
2. Must read credentials from `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars
3. Must complete in under 20 minutes on 2 vCPU / 8 GB RAM (no GPU)

**File:** `inference.py` (root of the repo)

```python
"""
inference.py — LLM-driven agent for GlobalProcurementEnv
Required by OpenEnv Hackathon. Must use OpenAI client. Runtime < 20 min.
"""

import os
import json
import requests
from openai import OpenAI

# ─── Read from environment variables (mandatory) ─────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")  # HF Space URL in prod
MAX_STEPS    = 15   # Keep low — must finish in <20 min across all 3 tasks
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an AI procurement officer. Each step you will receive the
current state of a supply chain simulation and must choose an action (integer 0-6):
0 = approve_cheapest, 1 = approve_fastest, 2 = approve_greenest,
3 = reject_all, 4 = negotiate_price, 5 = split_order, 6 = escalate.

Respond with ONLY a single integer (0-6). No explanation. No other text."""


def choose_action(observation: dict, task_id: int) -> int:
    """Uses the LLM to select an action given the current observation."""
    max_action = {1: 3, 2: 5, 3: 6}[task_id]

    user_msg = f"""Current procurement state:
- Step: {observation.get('step', 0)}
- Budget remaining: ${observation.get('budget_remaining', 0):,.0f}
- Violations so far: {observation.get('policy_violations_this_episode', 0)}
- Active disruptions: {observation.get('active_disruptions', [])}
- Available suppliers: {len([s for s in observation.get('suppliers', []) if s.get('available')])}

Choose action 0-{max_action}. Reply with ONE integer only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=5,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        action = int(raw)
        return max(0, min(action, max_action))  # clamp to valid range
    except Exception:
        return 0  # fallback to approve_cheapest on any error


def run_task_with_llm(task_id: int, seed: int = 42) -> float:
    """Runs one episode using the LLM and returns the grader score."""
    reset_resp = requests.post(f"{ENV_URL}/reset", json={"task": task_id, "seed": seed})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        action = choose_action(obs, task_id)
        step_resp = requests.post(f"{ENV_URL}/step", json={"action": action})
        step_resp.raise_for_status()
        result = step_resp.json()
        done = result.get("done", False)
        obs = result
        steps += 1

    state = requests.get(f"{ENV_URL}/state").json()
    return state.get("grader_score", 0.0) or 0.0


if __name__ == "__main__":
    print("=== GlobalProcurementEnv — LLM Inference ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_URL}\n")

    for task_id in [1, 2, 3]:
        score = run_task_with_llm(task_id=task_id, seed=42)
        print(f"Task {task_id}: LLM agent score = {score:.4f}")
```

**Why `MAX_STEPS = 15`?** Each LLM call takes ~2–5 seconds. With 3 tasks × 15 steps = 45 LLM calls, the worst-case runtime is around 3–4 minutes — well under the 20-minute limit. Setting it higher risks timeout disqualification.

### ✅ Pitstop Audit — Phase 1 (inference.py)

Test locally against the running FastAPI server (you don't need real LLM credentials for the structure test):

```bash
# Quick structural test — confirms the file runs without crashing on import
python -c "import inference; print('✅ inference.py imports cleanly')"

# Test with a real call (need ENV_URL pointing at local server)
ENV_URL=http://localhost:7860 \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
HF_TOKEN=your_token \
python inference.py
```

Pass condition: the script prints 3 lines of scores and exits cleanly. No HTTP 500 errors. No Python exceptions. **If `inference.py` crashes, fix it — this file's absence or failure is a disqualification.**

---

## Phase 2 — Write the Baseline Random Agent (`baseline.py`)

**What you're building:** A simple script that plays all 3 tasks using a completely random strategy — just picks a random action every step. This creates the "floor" score — the performance you'd get from a blindly random agent. Your trained PPO model on Day 4 must beat this floor.

**File:** `baseline.py` (root of the repo)

```python
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
    return score

if __name__ == "__main__":
    print("Running baseline random agent...")
    for task in [1, 2, 3]:
        score = run_task(task_id=task, seed=42)
        print(f"  Task {task}: baseline score = {score:.4f}")
```

**Why does this matter for the judges?** The judging system calls `baseline.py` to verify that the environment returns meaningful, non-trivial scores. If this script crashes or prints `0.0000` for all tasks, the submission is flagged.

### ✅ Pitstop Audit — Phase 1

Run: `python baseline.py` against the locally running FastAPI server.

Expected output looks like:
```
Running baseline random agent...
  Task 1: baseline score = 0.3421
  Task 2: baseline score = 0.2187
  Task 3: baseline score = 0.1654
```

Pass conditions: all 3 tasks print a float between 0 and 1, and no HTTP exceptions are raised. **If you get HTTP 500 errors, alert Jamal immediately — that's his side to fix.**

---

## Phase 3 — Write the Gymnasium Wrapper (`gym_wrapper.py`)

**What you're building:** A thin wrapper that makes your environment speak the Gymnasium protocol. Stable Baselines 3 (SB3) — the library you'll use to train PPO — doesn't know about your custom environment. It only knows how to train agents that follow the Gymnasium standard (`reset()` → returns `(obs, info)`, `step(action)` → returns `(obs, reward, terminated, truncated, info)`). This wrapper is the translator.

**File:** `env/gym_wrapper.py`

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from env.global_procurement_env import GlobalProcurementEnv

class ProcurementGymWrapper(gym.Env):
    """
    Gymnasium wrapper around GlobalProcurementEnv.
    This lets SB3's PPO train on our custom environment.
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
        obs_dict = self.env.reset(task=self.task, seed=self.seed)
        return self._dict_to_array(obs_dict), {}

    def step(self, action):
        result = self.env.step(int(action))
        obs_array = self._dict_to_array(result)
        reward = float(result["reward"])
        terminated = bool(result["done"])
        truncated = False
        return obs_array, reward, terminated, truncated, {}

    def _dict_to_array(self, obs: dict) -> np.ndarray:
        """Converts the JSON observation dict into a normalised float32 array."""
        MAX_BUDGET = 200000.0
        return np.array([
            obs.get("budget_remaining", 0.0) / MAX_BUDGET,
            obs.get("inventory", {}).get("steel", 0.0) / 1000.0,
            obs.get("inventory", {}).get("chips", 0.0) / 1000.0,
            obs.get("inventory", {}).get("fabric", 0.0) / 1000.0,
            obs.get("lead_days", 0) / 30.0,
            obs.get("carbon", 0.0) / 50.0,
            obs.get("violations", 0) / 10.0,
            obs.get("step", 0) / 100.0,
        ], dtype=np.float32)
```

**Why normalise to [0, 1]?** PPO uses a neural network internally, and neural networks train much better when all inputs are on the same scale. A raw budget of `187000.0` next to a violations count of `2` would make the network unstable. Dividing by the max possible value of each field keeps everything between 0 and 1.

### ✅ Pitstop Audit — Phase 2

```python
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
```
**`check_env` passes without errors and obs shape is (8,) → Phase 2 complete.**

---

## Phase 4 — Write the Training Script (`train.py`)

**What you're building:** The script that uses SB3's PPO algorithm to train an RL agent on Task 1. PPO (Proximal Policy Optimisation) is a state-of-the-art algorithm that learns by trying many actions, receiving rewards, and gradually improving its policy.

**File:** `train.py` (root of the repo)

```python
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
    print(f"✅ Model saved to {save_path}.zip")
    env.close()

if __name__ == "__main__":
    train_task(task_id=1, total_timesteps=50000, save_path="models/task1_ppo")
    # Task 2 and 3 training happens on Day 4
```

**How long will training take?** At 50,000 timesteps on a CPU, expect 5–15 minutes for Task 1. This is normal. Let it run.

### ✅ Pitstop Audit — Phase 3

Run `python train.py` and watch the output. SB3 will print training logs that look like:

```
------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 14.3          |
|    ep_rew_mean          | 0.427         |
| time/                   |               |
|    fps                  | 312           |
------------------------------------------
```

Pass conditions: the script runs to completion, `models/task1_ppo.zip` exists, and the file is under 100MB (`ls -lh models/task1_ppo.zip`).

---

## Phase 5 — Write the Trained Agent Runner (`trained_agent.py`)

**What you're building:** A script that loads your saved PPO models and runs them for 5 episodes each, printing average scores. This is how you (and the judges) verify that trained agents actually perform better than random.

**File:** `trained_agent.py` (root of the repo)

```python
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
```

### ✅ Pitstop Audit — Phase 4 (Day 3 Critical Gate)

Run `python trained_agent.py`. The trained agent's average reward must be **higher than the baseline score** you printed in Phase 1 for Task 1. If the trained agent scores the same or lower than random, something is wrong — either the reward function isn't giving the agent enough signal, or training ran for too few steps.

If scores are equal: increase `total_timesteps` to 100,000 in `train.py` and re-run. This is normal on first attempts.

**Trained score > random baseline score → Day 3 critical gate cleared.**
