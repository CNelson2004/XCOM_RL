from torch.utils.data import Dataset, DataLoader

# Dataset that wraps memory for a dataloader
class RLDataset(Dataset):
  def __init__(self, data):
    super().__init__()
    self.data = []
    for d in data:
      self.data.append(d)

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)


def calculate_return_xcom(memory, rollout, gamma):
  """Return memory with calculated return in experience tuple

    Args:
        memory (list): (state, action, action_dist, mask, return) tuples
        rollout (list): (state, action, action_dist, mask, reward) tuples from last rollout
        gamma (float): discount factor

    Returns:
        list: memory updated with (state, action, action_dist, mask, return) tuples from rollout
  """
  # Calculating R-hat: R-hat = sum(gamma^(i-t)*r)
  # Note: H is the index of the last tuple in rollout
  # t is the position in the rollout list
  H = len(rollout)-1
  for t, tup in enumerate(rollout):
    R_list = []
    for i in range(t,H+1): #+1 because ending range is exclusive
      r = rollout[i][-1]
      R_list.append(gamma**(i-t)*r)
    R_t = sum(R_list)
    # updating memory at location
    s,a,ad,mask,_ = tup
    memory.append((s,a,ad,mask,R_t))
  return memory


def calculate_return_cartpole(memory, rollout, gamma):
  """Return memory with calculated return in experience tuple

    Args:
        memory (list): (state, action, action_dist, return) tuples
        rollout (list): (state, action, action_dist, reward) tuples from last rollout
        gamma (float): discount factor

    Returns:
        list: memory updated with (state, action, action_dist, return) tuples from rollout
  """
  # Calculating R-hat: R-hat = sum(gamma^(i-t)*r)
  # Note: H is the index of the last tuple in rollout
  # t is the position in the rollout list
  H = len(rollout)-1
  for t, tup in enumerate(rollout):
    R_list = []
    for i in range(t,H+1): #+1 because ending range is exclusive
      r = rollout[i][3]
      R_list.append(gamma**(i-t)*r)
    R_t = sum(R_list)
    # updating memory at location
    s,a,ad,_ = tup
    memory.append((s,a,ad,R_t))
  return memory
