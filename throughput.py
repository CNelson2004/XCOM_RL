"""Measure how many environment steps per second this computer can run.

It times random legal play in:
  - one plain environment (the baseline),
  - SyncVectorEnv with 4 copies,
  - AsyncVectorEnv with 1, 4, 6, 7, and 8 copies (each copy in its own process).
Pick the AsyncVectorEnv size where adding copies stops helping much, and leave a core or two free
for the training process itself.
"""
import os
from time import perf_counter

import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, AutoresetMode
from gymnasium.wrappers import FlattenObservation

from env.xcom_sim import XcomEnv


def make_env():
    # The same environment the training code uses: flattened observation.
    return FlattenObservation(XcomEnv())


def random_actions(masks, rng):
    """One random legal action per copy. masks has one row per copy."""
    return np.array([rng.choice(np.flatnonzero(row)) for row in masks])


def time_single(steps=10_000):
    """Baseline: one environment, no vector wrapper."""
    env, rng = make_env(), np.random.default_rng(0)
    obs, info = env.reset(seed=0)
    start = perf_counter()
    for _ in range(steps):
        action = rng.choice(np.flatnonzero(info["action_mask"]))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:               # a plain environment does not reset itself
            obs, info = env.reset()
    seconds = perf_counter() - start
    env.close()
    return steps / seconds


def time_vector(kind, num, vector_steps=2_000):
    """kind is SyncVectorEnv or AsyncVectorEnv; num is the number of copies."""
    envs = kind([make_env for _ in range(num)], autoreset_mode=AutoresetMode.SAME_STEP)
    rng = np.random.default_rng(0)
    obs, info = envs.reset(seed=0)               # setup and first reset stay outside the timer
    for _ in range(20):                          # short warm-up so every process is running
        obs, reward, terminated, truncated, info = envs.step(random_actions(info["action_mask"], rng))
    start = perf_counter()
    for _ in range(vector_steps):
        obs, reward, terminated, truncated, info = envs.step(random_actions(info["action_mask"], rng))
    seconds = perf_counter() - start
    envs.close()                                 # shut down the background processes
    return vector_steps * num / seconds          # every envs.step advances all copies by one step


if __name__ == "__main__":
    print(f"CPU threads reported by the system: {os.cpu_count()}\n")
    results = [("1 plain environment", time_single())]
    results.append(("Sync, 4 copies", time_vector(SyncVectorEnv, 4)))
    for num in (1, 4, 6, 7, 8):
        results.append((f"Async, {num} cop{'y' if num == 1 else 'ies'}", time_vector(AsyncVectorEnv, num)))

    baseline = results[0][1]
    print(f"{'Setup':22s} {'steps/second':>14s} {'vs. 1 plain':>12s}")
    for name, rate in results:
        print(f"{name:22s} {rate:14,.0f} {rate / baseline:11.1f}x")

    best_name, best_rate = max(results, key=lambda r: r[1])
    print(f"\nFastest setup: {best_name} ({best_rate:,.0f} steps/second).")
    if best_name.startswith("Async"):
        print("If a smaller async setup is within about 10% of it, prefer the smaller one:")
        print("the training process needs a free core too.")
    else:
        print("Async never beat running everything in one process, so the cost of passing data")
        print("between processes outweighs the parallel work: train with SyncVectorEnv.")

# Note: For me, I will be using 6 environments on Async
