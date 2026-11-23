import torch
import torch.nn as nn
import numpy as np
from itertools import chain

device = torch.device("cpu")

def learn_ppo_xcom(optim, policy, value, memory_dataloader, epsilon, policy_epochs, norm_epsilon, max_grad_norm, entropy_coef):
  """Implement PPO policy and value network updates. Iterate over your entire
     memory the number of times indicated by policy_epochs.

    Args:
        optim (Adam): value and policy optimizer
        policy (PolicyNetwork): Policy Network
        value (ValueNetwork): Value Network
        memory_dataloader (DataLoader): dataloader with (state, action, action_dist, mask, return) tensors
        epsilon (float): trust region
        policy_epochs (int): number of times to iterate over all memory
  """
  # Calculating PolicyLoss, ValueLoss, and Loss
  # Also, calculate A-hat (when calculating use '.detach()')
  # pi'(a|s) is calculated
  # Note: ValueNetwork's forward returns value (tensor): estimated value, size (batch)

  # A-hat = R-hat - V(s)
  # ValueLoss = 1/N * sum((R-hat - V(s))**2)
  # PolicyLoss = -1/N * sum(min( currentPolicy(a|s)/dataPolicy(a|s) * A-hat, clip( currentPolicy(a|s)/dataPolicy(a|s), 1 - epsilon, 1 + epsilon) * A-hat ))
  # Loss = PolicyLoss + ValueLoss

  # clip function is torch.clamp(x, min, max)

  entropies = []
  value_losses = []
  policy_losses = []
  KLs = []
  norms = []
  policy_norms = []
  value_norms = []

  epoch = 0
  while epoch < policy_epochs:
    for mem in memory_dataloader:
      #getting advantage
      state,action,action_dist,mask,advantage,target = mem
      # making sure everything is on the correct device (and correct type)
      state = state.float().to(device)
      action = action.to(device)
      action_dist = action_dist.float().to(device)
      mask = mask.bool().to(device)
      advantage = advantage.float().to(device)
      target = target.float().to(device)
      # making sure advantage gets detached
      A_hat = advantage
      # normalizing advantage
      A_hat = (A_hat - torch.mean(A_hat))/(torch.std(A_hat) + norm_epsilon) # eps is here to prevent the bottom from being zero
      #Calculating PolicyLoss
      dist, log_dist = apply_mask_softmax(policy(state),mask)
      curr = dist.gather(1,action.unsqueeze(1)).squeeze(1) #Using gather to get one value from the row
      old = action_dist.gather(1,action.unsqueeze(1)).squeeze(1)
      # kl measures how much one epoch's updates changed the policy
      with torch.no_grad():
        ratio = curr/old
        log_ratio = torch.log(curr) - torch.log(old)
        kl = (ratio - 1) - log_ratio
        avg_kl = torch.mean(kl)
      KLs.append(avg_kl.item())
      PolicyLoss = -torch.mean(torch.min((curr/old)*A_hat, torch.clamp(curr/old, 1-epsilon, 1+epsilon)*A_hat))
      policy_losses.append(PolicyLoss.item())
      #Calculating ValueLoss
      # value(state) has shape (batch,1). It needs to be (batch,) to match target
      ValueLoss = torch.mean((target - value(state).squeeze(1))**2)
      value_losses.append(ValueLoss.item())
      #CalculatingLoss
      # calculating entropy
      entropy = -torch.sum(dist * log_dist, dim=1)
      avg_entropy = torch.mean(entropy)
      entropies.append(avg_entropy.item())
      Loss = PolicyLoss + ValueLoss - entropy_coef*avg_entropy
      #Updating
      optim.zero_grad()
      Loss.backward()
      # clamping gradient (Must be done after zero_grad and .backward)
      # Note: logging gradient norm to see if clipping is doing its job
      policy_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),max_grad_norm)
      policy_norms.append(policy_norm.item())
      value_norm = torch.nn.utils.clip_grad_norm_(value.parameters(),max_grad_norm)
      value_norms.append(value_norm.item())
      norm = float(np.sqrt(policy_norm.item()**2 + value_norm.item()**2))
      norms.append(norm)
      # continuing with updating
      optim.step()
    epoch += 1
  # TODO: Add guard in case list is empty (no batch was processed. (This could be due to too few steps in a battle or something else))
  return np.mean(entropies), np.mean(norms), np.max(norms), np.mean(policy_norms), np.mean(value_norms), np.mean(value_losses), np.mean(policy_losses), np.mean(KLs)
    

def apply_mask_softmax(logits,masks):
  # giving illegal actions a large negative score
  logits = logits.masked_fill(~masks,-999999999)
  # applying softmax
  dist = torch.softmax(logits,dim=1)
  log_dist = torch.log_softmax(logits,dim=1)
  return dist, log_dist


def get_action_ppo_xcom(network, obs, masks, is_training=True):
  """Sample action from the distribution obtained from the policy network

    Args:
        network (PolicyNetwork): Policy Network
        obs (np-array): current state, size (obs_size,6)

    Returns:
        np-array: 6 actions sampled from output distribution of policy network 
        np-array: output distributions of policy network (6,13)
  """
  # pi(a|s) calculated
  # Note: forward of PolicyNetwork returns action_dist (tensor): probability distribution over actions (batch x action_size)

  # Input size: (6,730), output is 6 actions and (6,2) array of action distributions

  # transforming observations and masks
  obs = torch.FloatTensor(obs).to(device)
  masks = torch.as_tensor(masks, dtype=torch.bool).to(device)

  # getting action_dist
  with torch.no_grad():
    action_logits = network(obs)
  action_dist,_ = apply_mask_softmax(action_logits,masks)

  if is_training:
    # sampling from distribution 
    actions = torch.multinomial(action_dist,1)
    actions = torch.flatten(actions) #flattening to shape (6,)
  else: # then we are evaluating
    # getting the action with the highest probability
    actions = torch.argmax(action_dist, dim=1) # Note: Gives same shape as multinomial after flattening
  return (actions.numpy(), action_dist.numpy()) # converting to numpy array (also can't convert while gpu)


def learn_ppo_cartpole(optim, policy, value, memory_dataloader, epsilon, policy_epochs):
  """Implement PPO policy and value network updates. Iterate over your entire
     memory the number of times indicated by policy_epochs.

    Args:
        optim (Adam): value and policy optimizer
        policy (PolicyNetwork): Policy Network
        value (ValueNetwork): Value Network
        memory_dataloader (DataLoader): dataloader with (state, action, action_dist, return) tensors
        epsilon (float): trust region
        policy_epochs (int): number of times to iterate over all memory
  """
  # Calculating PolicyLoss, ValueLoss, and Loss
  # Also, calculate A-hat (when calculating use '.detach()')
  # pi'(a|s) is calculated
  # Note: ValueNetwork's forward returns value (tensor): estimated value, size (batch)

  # A-hat = R-hat - V(s)
  # ValueLoss = 1/N * sum((R-hat - V(s))**2)
  # PolicyLoss = -1/N * sum(min( currentPolicy(a|s)/dataPolicy(a|s) * A-hat, clip( currentPolicy(a|s)/dataPolicy(a|s), 1 - epsilon, 1 + epsilon) * A-hat ))
  # Loss = PolicyLoss + ValueLoss

  # clip function is torch.clamp(x, min, max)

  epoch = 0
  while epoch < policy_epochs:
    for mem in memory_dataloader:
      #getting R-hat_t
      state,action,action_dist,R_hat = mem
      # making sure everything is on the correct device (and correct type)
      state = state.float().to(device)
      action = action.to(device)
      action_dist = action_dist.float().to(device)
      R_hat = R_hat.float().to(device) # using .float() because it plays nicer than the FloatTensor I used with dqn
      #Calculating A-hat_t
      V = value(state).squeeze(1)
      # making sure advantage gets detached
      A_hat = R_hat - V.detach()
      #Calculating PolicyLoss
      curr = policy(state).gather(1,action.unsqueeze(1)).squeeze(1) #Using gather to get one value from the row
      old = action_dist.gather(1,action.unsqueeze(1)).squeeze(1)
      PolicyLoss = -torch.mean(torch.min((curr/old)*A_hat, torch.clamp(curr/old, 1-epsilon, 1+epsilon)*A_hat))
      #Calculating ValueLoss
      ValueLoss = torch.mean((R_hat - V)**2)
      #CalculatingLoss
      Loss = PolicyLoss + ValueLoss
      #Updating
      optim.zero_grad()
      Loss.backward()
      optim.step()
    epoch += 1


def get_action_ppo_cartpole(network, state):
  """Sample action from the distribution obtained from the policy network

    Args:
        network (PolicyNetwork): Policy Network
        state (np-array): current state, size (state_size,4)

    Returns:
        np-array: 4 actions sampled from output distribution of policy network 
        np-array: output distributions of policy network (4,2)
  """
  # pi(a|s) calculated
  # Note: forward of PolicyNetwork returns action_dist (tensor): probability distribution over actions (batch x action_size)

  # Input size: (4,4), output is 4 actions and (4,2) array of action distributions

  state = torch.FloatTensor(state).to(device)
  # getting action_dist
  with torch.no_grad():
    action_dist = network(state)
  # sampling from distribution (drawing from probabilities - from research seems like torch.multinomial or distributions.Categorical could work)
  actions = torch.multinomial(action_dist,1)
  actions = torch.flatten(actions) #flattening to shape (4,)
  return (actions.numpy(), action_dist.numpy()) # converting to numpy array (also can't convert while gpu)
