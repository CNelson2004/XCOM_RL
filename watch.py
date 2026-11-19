import shutil
import glob
import io
import base64
import time
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from IPython.display import HTML
from IPython import display as ipythondisplay
from gymnasium.wrappers import FlattenObservation

from env.xcom_sim import XcomEnv, format_event
from agents.checkpoints import load_checkpoint
from agents.ppo import get_action_ppo_cartpole
from evaluate import learning_agent

ENV_KEYS = ("size", "n_soldiers", "n_aliens", "alien_hp", "wall_segments", "max_turns")


def wrap_env(env):
  env = RecordVideo(env, './video')
  return env

def show_video_cartpole():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else:
    print("Could not find video")

# code for recording a video that can be saved and watched
def record_episode_cartpole(policy_network):
  shutil.rmtree("video", ignore_errors=True)   # delete videos from earlier runs, so show_video() plays the new one
  env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=200)
  env = wrap_env(env)                           # the lab's RecordVideo helper; works on a single env
  state, _ = env.reset()
  done, total = False, 0
  while not done:
    actions, _ = get_action_ppo_cartpole(policy_network, state[None, :])   # batch of one: shape (1, 4)
    state, reward, terminated, truncated, _ = env.step(int(actions[0]))
    total += reward
    done = terminated or truncated
  env.close()                                   # closing is what writes the .mp4 file
  print("Episode reward:", total)

# code for watching video live
def watch_episode_cartpole(policy_network, episodes=1):
  env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=200)
  for _ in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
      actions, _ = get_action_ppo_cartpole(policy_network, state[None, :])   # batch of one
      state, reward, terminated, truncated, _ = env.step(int(actions[0]))
      done = terminated or truncated
  env.close()   # closes the window

def watch(check_path, battles=3, first_seed=2_000_000, step_through=True, delay=0.6):
    """Play `battles` battles with the saved agent, printing the map, each action and what happened.

    step_through=True waits for Enter after every action; False plays on its own, pausing `delay` seconds.
    """
    checkpoint = load_checkpoint(check_path)
    config = checkpoint["config"]
    # environment settings from the checkpoint (works whether they are under "env" or stored flat)
    env_kwargs = config.get("env_kwargs") or {k: config[k] for k in ENV_KEYS if k in config}
    policy = learning_agent(checkpoint)

    env = FlattenObservation(XcomEnv(**env_kwargs))   # the network sees the same flat observation as in training
    game = env.unwrapped                               # the XcomEnv inside, for render() and describe_action()

    for b in range(battles):
        obs, info = env.reset(seed=first_seed + b)
        done = False
        print(f"\n========== battle {b + 1} of {battles} ==========")
        while not done:
            print(game.render())
            action = policy(obs, info["action_mask"])
            print(f">> {game.describe_action(action, with_chance=True)}")   # chance shown before acting
            obs, reward, terminated, truncated, info = env.step(action)
            for e in info["events"]:
                print("   " + format_event(e))
            done = terminated or truncated
            if step_through and not done:
                input("   (Enter for the next action) ")
            elif not step_through:
                time.sleep(delay)
        print(game.render())
        print(f"RESULT: {info['result'].upper()}")
        if step_through and b < battles - 1:
            input("(Enter for the next battle) ")


if __name__ == "__main__":
    watch("runs/checkpoint_xcom_lr_decay.pt")
