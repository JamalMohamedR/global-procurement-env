import runpy
import sys

with open("eval_results.txt", "w") as f:
    sys.stdout = f
    sys.stderr = f
    try:
        runpy.run_path("trained_agent.py", run_name="__main__")
    except Exception as e:
        import traceback
        traceback.print_exc(file=f)
