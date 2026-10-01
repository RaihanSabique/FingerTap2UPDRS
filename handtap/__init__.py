"""HandTap2PDScore: finger-tapping video -> hand keypoints -> kinematics -> PD / MDS-UPDRS 3.4 score."""
from pathlib import Path

__version__ = "0.1.0"

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
# Weights live next to the package so the same layout works on the cluster and in a HF Space.
MODELS = Path(__import__("os").environ.get("HANDTAP_MODELS", ROOT / "models"))


def available_cpus():
    """CPUs this process may actually use. os.cpu_count() reports the host inside containers
    (e.g. dozens on a 2-vCPU Hugging Face Space), and sizing ONNX Runtime thread pools from
    it oversubscribes the CPU quota and slows inference by an order of magnitude."""
    import math
    import os
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit():
        n = min(n, int(os.environ["SLURM_CPUS_PER_TASK"]))
    try:  # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            n = min(n, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        try:  # cgroup v1
            quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if quota > 0:
                n = min(n, math.ceil(quota / period))
        except (OSError, ValueError):
            pass
    return max(1, n)
