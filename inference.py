"""
inference.py — LLM-driven agent for GlobalProcurementEnv
Required by OpenEnv Hackathon. Must use OpenAI client. Runtime < 20 min.
"""

import os
import json
import requests
from openai import OpenAI

# # ─── Read from environment variables (mandatory) ─────────────────────────────
# API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
# MODEL_NAME   = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
# ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")  # HF Space URL in prod
# MAX_STEPS    = 15   # Keep low — must finish in <20 min across all 3 tasks
# # ─────────────────────────────────────────────────────────────────────────────
# ─── Read from environment variables (mandatory) ─────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
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
