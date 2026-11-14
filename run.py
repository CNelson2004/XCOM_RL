# run.py: train, save the network, then record one episode
import os
from train import ppo_main_cartpole, ppo_main_xcom
from watch import record_episode
from plots import plot_history

# this is to check that moved code works with cartpole
# also to practice making a checkpoint for a network

def run(is_XCOM = True):

    if is_XCOM:
        results_ppo, policy_network, history = ppo_main_xcom()
        # plotting what we got (all saving already done in ppo_main)
        plot_history(history, "runs/training_curves.png",
        baselines={"random": 0.126, "scripted (advance)": 0.166, "scripted (hold)": 0.29})

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
# To load in network later (would need to be used inside ppo_main)

checkpoint = load_checkpoint("runs/checkpoint_xcom_final.pt")
config = checkpoint["config"]
policy_network = PolicyNetwork_Xcom(config["obs_size"], config["action_size"], config["hidden_size"])
policy_network.load_state_dict(checkpoint["policy"])
policy_network.eval() 

# how to plot from a saved checkpoint
plot_history("runs/checkpoint_xcom_epoch200.pt", "runs/epoch200_curves.png")
'''
