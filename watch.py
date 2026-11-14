import shutil
import glob
import io
import base64
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from IPython.display import HTML
from IPython import display as ipythondisplay

from agents.ppo import get_action_ppo_cartpole


def show_video():
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

def wrap_env(env):
  env = RecordVideo(env, './video')
  return env


# code for recording a video that can be saved and watched
def record_episode(policy_network):
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
def watch_episode(policy_network, episodes=1):
  env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=200)
  for _ in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
      actions, _ = get_action_ppo(policy_network, state[None, :])   # batch of one
      state, reward, terminated, truncated, _ = env.step(int(actions[0]))
      done = terminated or truncated
  env.close()   # closes the window
