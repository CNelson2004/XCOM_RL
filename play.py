"""Play a battle by hand. Run from the project's top folder: python play.py"""
import numpy as np
from env.xcom_sim import XcomEnv, format_event


def show_legal(env, mask):
    for a in np.flatnonzero(mask):
        print(f"  {a:2d}: {env.describe_action(a, with_chance=True)}")


def play(seed=1, actions=None):
    """actions: optional list of action numbers to replay first; after that it asks you."""
    env = XcomEnv()
    obs, info = env.reset(seed=seed)
    done, step = False, 0
    while not done:
        print(env.render())
        print("legal actions:")
        show_legal(env, info["action_mask"])
        if actions and step < len(actions):          # replaying a list of moves
            a = actions[step]; print(f"action> {a}")
        else:
            a = int(input("action> "))
        obs, r, term, trunc, info = env.step(a)
        print(f">> {env.describe_action(a)}")
        for e in info["events"]:
            print("   " + format_event(e))
        print()
        done, step = term or trunc, step + 1
    print(env.render())
    print("RESULT:", info["result"])


if __name__ == "__main__":
    play()
