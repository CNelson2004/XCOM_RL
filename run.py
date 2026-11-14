# run.py: train, save the network, then record one episode
import os
import torch
from train import ppo_main_cartpole, ppo_main_xcom
from watch import record_episode

# this is to check that moved code works with cartpole
# also to practice making a checkpoint for a network

def run(is_XCOM = True):

    if is_XCOM:
        results_ppo, policy_network, history = ppo_main_xcom()
        os.makedirs("runs", exist_ok=True)
        print(history)
        # saving network
        torch.save(policy_network.state_dict(), "runs/policy_xcom.pt")   # saves the learned weights
        print("Saved network to runs/policy_xcom.pt")

    else:
        results_ppo, policy_network = ppo_main_cartpole()
        os.makedirs("runs", exist_ok=True)
        # saving network
        #torch.save(policy_network.state_dict(), "runs/policy_cartpole.pt")   # saves the learned weights
        #print("Saved network to runs/policy_cartpole.pt")
    
        # video of training to watch
        record_episode(policy_network)
        print("Video saved in the video/ folder")

if __name__ == "__main__":
    #run(is_XCOM = False) # for running cartpole
    run()

'''
# To load in network later

from agents.networks import PolicyNetwork

policy_network = PolicyNetwork(state_size=4, action_size=2)
policy_network.load_state_dict(torch.load("runs/policy_cartpole.pt"))
policy_network.eval()   # switches the network into "using it, not training it" mode

'''