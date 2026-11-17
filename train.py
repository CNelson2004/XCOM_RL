import torch
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from gymnasium.vector import AutoresetMode
from itertools import chain
from tqdm import tqdm
import numpy as np

from agents.buffer import calculate_return_cartpole, calculate_return_xcom, RLDataset
from agents.networks import PolicyNetwork_Cartpole, PolicyNetwork_Xcom, ValueNetwork_Cartpole, ValueNetwork_Xcom
from agents.ppo import get_action_ppo_cartpole, learn_ppo_cartpole, get_action_ppo_xcom, learn_ppo_xcom
from agents.checkpoints import save_checkpoint, load_checkpoint

from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from gymnasium.wrappers import FlattenObservation

from env.xcom_sim import XcomEnv


def ppo_main_xcom(epochs=3,save_rate=100,size=12,n_soldiers=2,n_aliens=2,alien_hp=30,wall_segments=None):
  # Hyper parameters
  lr = 3e-4
  gamma = 0.99
  batch_size = 256
  epsilon = 0.2
  policy_epochs = 5
  hidden_size = 128

  norm_epsilon = 1e-8
  max_grad_norm = 0.5
  entropy_coef = 0.01

  battles = 100
  num_steps = 2000
  max_turns=30

  obs_size = 5 * size * size + 10 #size of total state model can see (flattened obs: grid: (12 x 12)*5 + unit: 10)
  action_size = 8 + 2 * n_aliens + 1 #number of actions that can be taken (8 types of movement, 2 shots for each alien, and passing turn)

  # Init environment
  # making the parallel XCOM environments
  num_envs = 6
  envs = AsyncVectorEnv(
      [lambda: FlattenObservation(XcomEnv(size=size, n_soldiers=n_soldiers, n_aliens=n_aliens, max_turns=max_turns, alien_hp=alien_hp, wall_segments=wall_segments)) for _ in range(num_envs)],
      autoreset_mode=AutoresetMode.SAME_STEP,
  )

  # setting config to be used for checkpoint
  env_kwargs = {"size": size, "n_soldiers": n_soldiers, "n_aliens": n_aliens,
              "alien_hp": alien_hp, "wall_segments": wall_segments, "max_turns" : max_turns}
  config = {"hidden_size": hidden_size, "lr": lr, "gamma": gamma, "epsilon": epsilon, "batch_size" : batch_size, "policy_epochs": policy_epochs, # hyperparameters
            "norm_epsilon" : norm_epsilon, "max_grad_norm" : max_grad_norm, "entropy_coef" : entropy_coef, "battles" : battles, "num_steps" : num_steps, # more hyperparameters
            "obs_size": obs_size, "action_size": action_size, # calculated vars
            "num_envs": num_envs, "env_kwargs" : env_kwargs} # environment vars

  # getting initial observations
  obs, _ = envs.reset(seed=0)       

  # Init networks
  policy_network = PolicyNetwork_Xcom(obs_size, action_size, hidden_size).cpu() #running on my laptop, so no cuda
  value_network = ValueNetwork_Xcom(obs_size, hidden_size).cpu()

  # Init optimizer
  optim = torch.optim.Adam(chain(policy_network.parameters(), value_network.parameters()), lr=lr)

  # Start main loop
  results_ppo = []
  loop = tqdm(total=epochs, position=0, leave=False)
  history = {"win rate": [], "loss rate": [], "timeout rate": [], "avg entropy": [], "avg grad norm": [], "max grad norm": [], "illegals" : []} # illegals should always be 0
  # Note: there is no inner episode loop because we are runnign multiple environments which can end at different times
  for epoch in range(epochs):
    # Reset environment before the epoch loop
    obs, info = envs.reset()
    masks = info["action_mask"]

    # setting variables for loop
    memory = []  # Reset memory every epoch
    total_rewards = []  # Calculate average episodic reward per epoch
    the_results = {"win": 0, "loss": 0, "timeout": 0}
    rollouts = [[] for i in range(num_envs)]
    cum_rewards = [0 for i in range(num_envs)]  # Track cumulative rewards
    illegals = 0
    episodes = 0
    
    # starting loop
    # when doing small battles on small maps you don't get enough data just through 100 battles, since so few steps are taken
    # thus we need to make sure enough steps are taken and enough battles are done to get enough data to learn from
    # each memory item is a step, thus we can count those

    while episodes < battles or len(memory) < num_steps:
      # Getting actions for each environment
      actions, action_dists = get_action_ppo_xcom(policy_network, obs, masks)

      # Taking steps for each environment
      next_obs, rewards, terminateds, truncateds, info = envs.step(actions)
      next_masks = info["action_mask"]

      # checking masking is working
      illegals += np.sum(info["illegal"])

      # storing steps
      for i in range(num_envs):
        # updating rollout
        rollouts[i].append((obs[i],actions[i],action_dists[i],masks[i],rewards[i]))
        # updating rewards
        cum_rewards[i] += rewards[i]

      # checking if done
      dones = [term or trunc for term,trunc in zip(terminateds,truncateds)]
      for i, don in enumerate(dones):
        if don:
          # cleaning up now that it is done
          # doing double check for results and illegals
          the_results[info["final_info"]["result"][i]] += 1
          illegals += int(info["final_info"]["illegal"][i])
          # rollout to memory
          memory = calculate_return_xcom(memory, rollouts[i], gamma)
          # recording cumulative rewards for episode
          total_rewards.append(cum_rewards[i])
          # clearing out rollout and cumulative rewards 
          rollouts[i] = []
          cum_rewards[i] = 0
          # incrementating counter 
          episodes += 1

      obs = next_obs
      masks = next_masks

    # Train
    dataset = RLDataset(memory) 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True) # dropping lat to not have weird edge case where std later is nan while normalizing.
    entropy, norm, max_norm = learn_ppo_xcom(optim, policy_network, value_network, loader, epsilon, policy_epochs, norm_epsilon, max_grad_norm,entropy_coef)

    #computing rates for this epoch
    win_rate = the_results["win"]/episodes
    loss_rate = the_results["loss"]/episodes
    timeout_rate = the_results["timeout"]/episodes
    # adding to history
    history["win rate"].append(win_rate)
    history["loss rate"].append(loss_rate)
    history["timeout rate"].append(timeout_rate)
    history["avg entropy"].append(float(np.round(entropy,2)))
    history["avg grad norm"].append(float(np.round(norm,2)))
    history["max grad norm"].append(float(np.round(max_norm,2)))
    history["illegals"].append(int(illegals))

    # Print results
    results_ppo.extend(total_rewards)  # Store rewards for this epoch
    loop.update(1)
    loop.set_description("Epochs: {} Win Rate: {}".format(epoch, win_rate))

    # making checkpoint 
    if (epoch+1) % save_rate == 0:
      path = f"runs/checkpoint_xcom_epoch{epoch + 1}.pt"
      save_checkpoint(path, policy_network, value_network, optim, epoch + 1, history, config)
      tqdm.write(f"Saved {path}")

  # making one last save
  save_checkpoint("runs/checkpoint_xcom_final.pt", policy_network, value_network, optim, epochs, history, config)
  # closing environments since training is done
  envs.close()

  return results_ppo, policy_network, history


# staying here so I can always check back
def ppo_main_cartpole(epochs=3):
  # Hyper parameters
  lr = 1e-3
  env_samples = 100
  gamma = 0.9
  batch_size = 256
  epsilon = 0.2
  policy_epochs = 5
  hidden_size = 8

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
  policy_network = PolicyNetwork_Cartpole(state_size, action_size,hidden_size).cpu() #running on my laptop, so no cuda
  value_network = ValueNetwork_Cartpole(state_size,hidden_size).cpu()

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
      actions, action_dists = get_action_ppo_cartpole(policy_network, states)

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
          memory = calculate_return_cartpole(memory, rollouts[i], gamma)
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
    learn_ppo_cartpole(optim, policy_network, value_network, loader, epsilon, policy_epochs)

    # Print results
    results_ppo.extend(total_rewards)  # Store rewards for this epoch
    loop.update(1)
    loop.set_description("Epochs: {} Reward: {}".format(epoch, results_ppo[-1]))

  # closing environments since training is done
  envs.close()

  return results_ppo, policy_network

