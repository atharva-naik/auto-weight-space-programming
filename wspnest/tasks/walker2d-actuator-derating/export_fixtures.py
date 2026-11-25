"""Export four trained Walker2d actors and calibration second moments."""
import argparse
from hashlib import sha256
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np

from model import CANARY, rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(a.study.resolve()))
    from search import Policy

    a.out.mkdir(parents=True, exist_ok=True)
    env = gym.make("Walker2d-v5")
    try:
        for i in range(1, 5):
            source = a.study / "extension/checkpoints" / f"Walker2d-v5_{i}.zip"
            policy = Policy(source)
            (w0, b0), (w1, b1) = policy.hidden
            actor = dict(w0=w0, b0=b0, w1=w1, b1=b1, head=policy.head)
            rows = np.array([h for seed in range(291000, 291006)
                             for h in rollout(env, actor, policy.head, 1., seed, trace=True)[1]])
            np.savez_compressed(a.out / f"actor-{i}.npz", **actor, cov=rows.T @ rows / len(rows),
                                source_sha256=sha256(source.read_bytes()).hexdigest(), canary=CANARY)
    finally:
        env.close()


if __name__ == "__main__":
    main()
