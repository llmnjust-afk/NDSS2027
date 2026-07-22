from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
from agentdojo.task_suite import get_suite
from agentdojo.benchmark import run_task_with_injection_tasks
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.logging import OutputLogger
from sok_ipl.native.pipeline_builder import build_pipeline
# Fix pydantic forward-ref for agentdojo 0.1.35 TaskResults
from agentdojo.functions_runtime import FunctionCall
import agentdojo.benchmark as _bm
try:
    _bm.TaskResults.model_rebuild()
except Exception as _e:
    print("rebuild note:", _e)

KEY='sk-9ZuUA9MWpglHgMJhKyBsPUDnGDm95ygy9yN4YqVoLc7GsRp0'
M='gpt-4o-mini-2024-07-18'
suite = get_suite("v1.2.2","workspace")
logdir = Path("/data/lab/NDSS2027/runs_smoke")
for d in ["none","task_shield"]:
    p = build_pipeline(d, M, KEY)
    attack = load_attack("important_instructions", suite, p)
    with OutputLogger(str(logdir)):
        ut, sec = run_task_with_injection_tasks(
            suite, p, suite.get_user_task_by_id("user_task_0"),
            attack, logdir, True, ["injection_task_0"], "v1.2.2")
    u = list(ut.values())[0]; s = list(sec.values())[0]
    print(f"{d:14s} utility={u}  security_breached={s}")
