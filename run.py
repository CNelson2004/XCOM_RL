# run.py: train, save the network, then record one episode
import os
from train import ppo_main_cartpole, ppo_main_xcom
from watch import record_episode_cartpole
from env.rewards import tu_reward, tu_and_damage_reward, damage_reward
from evaluate import evaluate_learner

# this is to check that moved code works with cartpole
# also to practice making a checkpoint for a network

def run(is_XCOM = True, epochs=500, battles=100, reward_fn = None, alien_behavior="hunt", save_rate=100, entropy_coef=0.01, size=12,n_soldiers=2,n_aliens=2,alien_hp=30,wall_segments=None):

    if is_XCOM:
        results_ppo, policy_network, history = ppo_main_xcom(epochs=epochs,battles=battles,reward_fn=reward_fn,alien_behavior=alien_behavior,save_rate=save_rate,entropy_coef=entropy_coef,size=size,n_soldiers=n_soldiers,n_aliens=n_aliens,alien_hp=alien_hp,wall_segments=wall_segments)
        # plotting done when we do evaluation, so it isn't done here
        # add env.render()?

    else:
        results_ppo, policy_network = ppo_main_cartpole()
        os.makedirs("runs", exist_ok=True)
        # saving network
        #torch.save(policy_network.state_dict(), "runs/policy_cartpole.pt")   # saves the learned weights
        #print("Saved network to runs/policy_cartpole.pt")
    
        # video of training to watch
        record_episode_cartpole(policy_network)
        print("Video saved in the video/ folder")

def run_and_evaluate(epochs=300,battles=100,reward_fn=None, alien_behavior="hunt",save_rate=50,name=None, n_battles=500, init_seed=1_000_000):
    run(epochs=epochs,battles=battles,save_rate=save_rate,reward_fn=reward_fn,alien_behavior=alien_behavior)
    evaluate_learner(name=name, n_battles=n_battles, init_seed=init_seed)


if __name__ == "__main__":
    #run(is_XCOM = False) # for running cartpole
    #run(epochs=300,reward_fn=tu_reward,save_rate=50)
    run_and_evaluate(epochs=200,battles=20,save_rate=25,reward_fn=damage_reward, alien_behavior="guard")
