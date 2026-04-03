# integration_test.py -- Phase 1: Full Integration Test (All 3 Tasks via FastAPI)
# Runs 10 full episodes per task, collecting grader scores.
# Pass conditions: no HTTP errors, all 30 calls complete, each task shows >=3 distinct scores.

import requests
import random
import sys
import os

# Force UTF-8 output so redirected output doesn't crash on Windows cp1252
os.environ['PYTHONIOENCODING'] = 'utf-8'

BASE_URL = "http://localhost:7860"

# Safety step limit — Task 3 has MAX_STEPS=100 but budget runs out much earlier.
# If the episode hasn't ended after this many steps, something may be wrong.
SAFETY_MAX_STEPS = 150


def run_episode(task_id: int, seed: int, max_actions: dict = {1: 4, 2: 6, 3: 7}) -> dict:
    """Runs one full episode via the API and returns the state at the end."""
    # Reset the environment for this episode
    resp = requests.post(f"{BASE_URL}/reset", json={"task": task_id, "seed": seed})
    if resp.status_code != 200:
        print(f"    [WARN] /reset failed (HTTP {resp.status_code}): {resp.text[:200]}")
        return {"steps": 0, "state": {}, "grader_score": None, "error": resp.text[:200]}

    done = False
    n_actions = max_actions[task_id]
    steps = 0

    while not done and steps < SAFETY_MAX_STEPS:
        action = random.randint(0, n_actions - 1)
        try:
            step_resp = requests.post(f"{BASE_URL}/step", json={"action": action})
        except requests.exceptions.ConnectionError as e:
            print(f"    [WARN] Connection lost at step {steps}: {e}")
            return {"steps": steps, "state": {}, "grader_score": None, "error": str(e)}

        if step_resp.status_code != 200:
            # A 400 here usually means "Episode has ended. Call /reset."
            # This is expected if a hard violation ended the episode on the previous step.
            # Just break out and read the final state.
            print(f"    [INFO] /step returned HTTP {step_resp.status_code} at step {steps} -- ending episode")
            break

        last_result = step_resp.json()
        done = last_result.get("done", False)
        steps += 1

    if steps >= SAFETY_MAX_STEPS:
        print(f"    [WARN] Hit safety limit ({SAFETY_MAX_STEPS} steps) for task {task_id} seed {seed}")

    # Retrieve the final state (includes grader_score if episode ended)
    try:
        state_resp = requests.get(f"{BASE_URL}/state")
        if state_resp.status_code == 200:
            state = state_resp.json()
        else:
            state = {}
    except requests.exceptions.ConnectionError:
        state = {}

    return {"steps": steps, "state": state, "grader_score": state.get("grader_score", None)}


# ── Main test loop ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 1 — Full Integration Test (10 episodes × 3 tasks)")
print("=" * 60)

# Quick health check first
try:
    health = requests.get(f"{BASE_URL}/health")
    if health.status_code == 200:
        print(f"[PASS] /health OK: {health.json()}")
    else:
        print(f"[FAIL] /health failed with HTTP {health.status_code}")
        sys.exit(1)
except requests.exceptions.ConnectionError:
    print(f"[FAIL] Cannot reach server at {BASE_URL} -- is uvicorn running?")
    sys.exit(1)

all_passed = True

for task_id in [1, 2, 3]:
    scores = []
    errors = 0
    print(f"\n--- Task {task_id} ---")
    for seed in range(10):
        result = run_episode(task_id=task_id, seed=seed)
        score = result["grader_score"]
        scores.append(score)
        if result.get("error"):
            errors += 1

    print(f"\nTask {task_id} scores across 10 seeds:")
    for i, s in enumerate(scores):
        print(f"  seed={i}: {s}")

    # Filter out None scores for uniqueness check
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        print(f"  [FAIL] No valid scores returned -- all were None")
        all_passed = False
        continue

    unique = len(set(str(round(s, 3)) for s in valid_scores))
    if unique < 3:
        print(f"  [WARN] Only {unique} unique score values -- grader may be too constant")
        all_passed = False
    else:
        print(f"  [PASS] {unique} distinct score values -- grader is working")

    if errors > 0:
        print(f"  [WARN] {errors}/10 episodes had HTTP errors")

print("\n" + "=" * 60)
if all_passed:
    print("[PASS] Phase 1 -- ALL TASKS PASSED")
else:
    print("[WARN] Phase 1 -- Some issues detected (see warnings above)")
print("=" * 60)
