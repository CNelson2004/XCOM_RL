# run.py: train, save the network, then record one episode
import os
from train import ppo_main_cartpole, ppo_main_xcom
from watch import record_episode

# this is to check that moved code works with cartpole
# also to practice making a checkpoint for a network

def run(is_XCOM = True, epochs=500, size=12,n_soldiers=2,n_aliens=2,alien_hp=30,wall_segments=None):

    if is_XCOM:
        results_ppo, policy_network, history = ppo_main_xcom(epochs,size=size,n_soldiers=n_soldiers,n_aliens=n_aliens,alien_hp=alien_hp,wall_segments=wall_segments)
        # plotting done when we do evaluation, so it isn't done here
        # add env.render()?

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
    run(epochs=200,size=8,n_soldiers=1,n_aliens=1,alien_hp=30,wall_segments=1)
