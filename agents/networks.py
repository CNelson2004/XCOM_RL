import torch.nn as nn
import torch

class PolicyNetwork_XcomCNN(nn.Module):
  def __init__(self, state_size, action_size, layer_size, map_size, cnn_hidden_size, mlp_hidden_size):
    super().__init__()
    # For 12 x 12 map_size would just be 12
    self.layer_size = layer_size
    self.map_size = map_size
    # Note: layer size should be five because the grid has five layers
    # cnn hidden size for this one should be about 32 (instead of 128 like it is for mlp)
    self.cnn = nn.Sequential(nn.Conv2d(in_channels=layer_size, out_channels=cnn_hidden_size, kernel_size=(3,3), padding=(1,1)), 
                                 nn.ReLU(), 
                                 nn.Conv2d(in_channels=cnn_hidden_size, out_channels=cnn_hidden_size, kernel_size=(3,3), padding=(1,1)), 
                                 nn.ReLU(), 
                                 nn.Conv2d(in_channels=cnn_hidden_size, out_channels=cnn_hidden_size, kernel_size=(3,3), padding=(1,1)),
                                 nn.ReLU())
                                 # ReLU added to end since it will be fed to mlp
                                 # CNN is extracting map features

    # This is the size of the flattened input (grid and units respectively)
    mlp_input = cnn_hidden_size * map_size * map_size + state_size - map_size * map_size * layer_size
    self.mlp = nn.Sequential(nn.Linear(in_features= mlp_input, out_features=mlp_hidden_size),
                             nn.ReLU(),
                             nn.Linear(in_features=mlp_hidden_size, out_features=action_size))
                             # Dense network is to extract action

  def forward(self, x):
    # split input into grid and unit features
    grid, unit = torch.tensor_split(x,[self.map_size * self.map_size * self.layer_size],dim=1)
    # reshaping flat grid inputs back into batches
    batch_grid = torch.reshape(grid,(-1,self.layer_size,self.map_size,self.map_size))
    # running grid through CNN for more info
    output = self.cnn(batch_grid)
    # flattening result
    flat_output = torch.flatten(output,start_dim=1)
    # joining unit features back onto grid
    new_x = torch.concat((flat_output,unit),dim=1)
    # getting action from MLP
    return self.mlp(new_x)


class ValueNetwork_XcomCNN(nn.Module):
  def __init__(self, state_size, layer_size, map_size, cnn_hidden_size, mlp_hidden_size):
    super().__init__()

    self.layer_size = layer_size
    self.map_size = map_size

    self.cnn = nn.Sequential(nn.Conv2d(in_channels=layer_size, out_channels=cnn_hidden_size, kernel_size=(3,3), padding=(1,1)), 
                                  nn.ReLU(), 
                                  nn.Conv2d(in_channels=cnn_hidden_size, out_channels=cnn_hidden_size, kernel_size=(3,3), padding=(1,1)), 
                                  nn.ReLU(), 
                                  nn.Conv2d(in_channels=cnn_hidden_size, out_channels=cnn_hidden_size, kernel_size=(3,3), padding=(1,1)),
                                  nn.ReLU())

    mlp_input = cnn_hidden_size * map_size * map_size + state_size - map_size * map_size * layer_size
    self.mlp = nn.Sequential(nn.Linear(in_features= mlp_input, out_features=mlp_hidden_size),
                              nn.ReLU(),
                              nn.Linear(in_features=mlp_hidden_size, out_features=1))

  def forward(self, x):
    grid, unit = torch.tensor_split(x,[self.map_size * self.map_size * self.layer_size],dim=1)
    batch_grid = torch.reshape(grid,(-1,self.layer_size,self.map_size,self.map_size))
    output = self.cnn(batch_grid)
    flat_output = torch.flatten(output,start_dim=1)
    new_x = torch.concat((flat_output,unit),dim=1)
    return self.mlp(new_x)


class PolicyNetwork_Xcom(nn.Module):
  def __init__(self, state_size, action_size, hidden_size):
    super().__init__()

    self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, action_size))
                              # removed softmax to return raw scores rather than probabilities

  def forward(self, x):
    """Get policy from state

      Args:
          state (tensor): current state, size (batch x state_size)

      Returns:
          action_scores (tensor): raw scores over actions (batch x action_size)
    """
    return self.net(x)


class ValueNetwork_Xcom(nn.Module):
  def __init__(self, state_size, hidden_size):
    super().__init__()

    self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, 1))

  def forward(self, x):
    """Estimate value given state

      Args:
          state (tensor): current state, size (batch x state_size)

      Returns:
          value (tensor): estimated value, size (batch)
    """
    return self.net(x)


class PolicyNetwork_Cartpole(nn.Module):
  def __init__(self, state_size, action_size, hidden_size):
    super().__init__()

    self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, action_size),
                             nn.Softmax(dim=1))

  def forward(self, x):
    """Get policy from state

      Args:
          state (tensor): current state, size (batch x state_size)

      Returns:
          action_dist (tensor): probability distribution over actions (batch x action_size)
    """
    return self.net(x)


class ValueNetwork_Cartpole(nn.Module):
  def __init__(self, state_size, hidden_size):
    super().__init__()

    self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, 1))

  def forward(self, x):
    """Estimate value given state

      Args:
          state (tensor): current state, size (batch x state_size)

      Returns:
          value (tensor): estimated value, size (batch)
    """
    return self.net(x)
