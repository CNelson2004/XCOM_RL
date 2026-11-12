import torch
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from gymnasium.vector import AutoresetMode
from itertools import chain
from tqdm import tqdm

from agents.buffer import calculate_return, RLDataset
from agents.networks import PolicyNetwork, ValueNetwork
from agents.ppo import get_action_ppo, learn_ppo


def ppo_main():
  # Hyper parameters
  lr = 1e-3
  epochs = 20
  env_samples = 100
  gamma = 0.9
  batch_size = 256
  epsilon = 0.2
  policy_epochs = 5

  # Init environment
  # I added this section so I can run 4 parallel environments (This was provided by AI because it deals with the environment)
  num_envs = 4
  envs = gym.make_vec(
      "CartPole-v1",
      num_envs=num_envs,
      vectorization_mode="sync",       # all 4 in one process
      max_episode_steps=200,           # replaces the lab's "cum_reward < 200" check
      vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
  )
  states, _ = envs.reset(seed=0)       # states.shape == (4, 4): one row per environment

  # Note: DQN was removed because we aren't using it
  state_size = 4
  action_size = 2
  #env = gym.make('CartPole-v1', render_mode="rgb_array") # This is left over from lab

  # Init networks
  policy_network = PolicyNetwork(state_size, action_size).cpu() #running on my laptop, so no cuda
  value_network = ValueNetwork(state_size).cpu()

  # Init optimizer
  optim = torch.optim.Adam(chain(policy_network.parameters(), value_network.parameters()), lr=lr)

  # Start main loop
  results_ppo = []
  loop = tqdm(total=epochs, position=0, leave=False)
  # Note: there is no inner episode loop because we are runnign multiple environments which can end at different times
  for epoch in range(epochs):
    # Reset environment before the epoch loop
    states, _ = envs.reset()

    # setting variables for loop
    memory = []  # Reset memory every epoch
    total_rewards = []  # Calculate average episodic reward per epoch
    rollouts = [[] for i in range(num_envs)]
    cum_rewards = [0 for i in range(num_envs)]  # Track cumulative rewards
    episodes = 0

    # starting loop
    while episodes < env_samples:
      # Getting actions for each environment
      actions, action_dists = get_action_ppo(policy_network, states)

      # Taking steps for each environment
      next_states, rewards, terminateds, truncateds, info = envs.step(actions)

      # storing steps
      for i in range(num_envs):
        # updating rollout
        rollouts[i].append((states[i],actions[i],action_dists[i],rewards[i]))
        # updating rewards
        cum_rewards[i] += rewards[i]

      # checking if done
      dones = [term or trunc for term,trunc in zip(terminateds,truncateds)]
      for i, don in enumerate(dones):
        if don:
          # cleaning up now that it is done
          # rollout to memory
          memory = calculate_return(memory, rollouts[i], gamma)
          # recording cumulative rewards for episode
          total_rewards.append(cum_rewards[i])
          # clearing out rollout and cumulative rewards 
          rollouts[i] = []
          cum_rewards[i] = 0
          # incrementating counter 
          episodes += 1

      states = next_states

    # Train
    dataset = RLDataset(memory) 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    learn_ppo(optim, policy_network, value_network, loader, epsilon, policy_epochs)

    # Print results
    results_ppo.extend(total_rewards)  # Store rewards for this epoch
    loop.update(1)
    loop.set_description("Epochs: {} Reward: {}".format(epoch, results_ppo[-1]))

  # closing environments since training is done
  envs.close()

  return results_ppo, policy_network

