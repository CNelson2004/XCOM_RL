import numpy as np
import glob, os, re
from tqdm import tqdm
from gymnasium.wrappers import FlattenObservation
from env.xcom_sim import XcomEnv
from agents.networks import PolicyNetwork_Xcom
from agents.checkpoints import load_checkpoint
from agents.ppo import get_action_ppo_xcom
from view.plots import plot_history

# evaluate the policy given to it
def evaluate(policy, n_battles=500, init_seed=1_000_000, flatten=True, desc=None, **env_kwargs):
    # Note: env_kwargs includes: map_size, num_soldiers, num_aliens, max_turns, etc.

    # setting up environment
    env = XcomEnv(**env_kwargs)
    # change into shape we need
    if flatten:
        env = FlattenObservation(env)
    # set vars
    results = []
    alive = []
    turns = []
    # start loop
    # progress bar over battles; only shown when a desc is given, and removed when finished (leave=False)
    # start battle loop
    for i in tqdm(range(n_battles), desc=desc, leave=False, disable=desc is None):
        # have a set seed (reset gives first observation and action mask)
        obs, info = env.reset(seed=init_seed + i)
        done = False
        # loop doing the battle (lop repeats until 'win' or 'lost' or hit term limit)
        while not done:
            # Note: reward won't be used because evaluation judges outcome, learning uses reward
            obs, reward, terminated, truncated, info = env.step(policy(obs, info["action_mask"]))
            # Note: Trunc indicates hitting turn limit, term inicates battle was won or lost (all aliens or soldiers killed)
            done = terminated or truncated
        # getting whether battle was won or lost or timed out
        results.append(info["result"])
        # getting number of soldiers alive
        alive.append(info["soldiers_alive"])
        # getting the turn the battle ended on (basically number of turns taken)
        turns.append(info["turn"])
    # win rate and its 95% margin of error (normal approximation for a yes/no outcome over n_battles)
    win_rate = results.count("win") / n_battles
    ci95 = 1.96 * np.sqrt(win_rate * (1 - win_rate) / n_battles)
    # returning win, loss, and timeout rates, the win rate's margin of error,
    # how many soldiers were alive (on average) and turns taken (on average)
    return dict(win_rate=win_rate, ci95=float(ci95), # win rate and a confidence interval (so we can check if we are in margin of error of baseline)
                loss_rate=results.count("loss") / n_battles, timeout_rate=results.count("timeout") / n_battles, # otheri nteresting rates
                soldiers_alive=float(np.mean(alive)), turns=float(np.mean(turns))) # other interesting metrics

def random_agent():
    # making agent that chooses actions at random so we have a baseline to beat
    # Note: doesn't learn, just a metric to use
    # setting rng
    # Note: rng must be set outside the policy itself so it uses the same rng each time
    rng = np.random.default_rng(0)
    # creating policy that chooses actions at random
    def random_policy(obs,mask):
        # Note: mask is action mask showing what move is legal
        # getting all indices where mask is True (action is allowed)
        # legal_actions = [i for i, m in enumerate(np.ravel(mask)) if m] 
        # Using numpy equivalent because it is faster 
        legal_actions = np.flatnonzero(mask)
        # choosing one of the allowed indexes at random
        # chosen = legal_actions[rng.integers(len(legal_actions))]
        # Again, using numpy version because faster. 
        chosen = rng.choice(legal_actions)
        # making sure index is int 
        return int(chosen)
    return random_policy

def scripted_agent(fixed_position=True):
    # Note: doesn't learn, just a metric to use

    def aggressive_policy(obs,mask):
        # Note: obs is a dict that contains "grid" and "unit" information
        # "unit" is an array that contains (in percentages) [0-9]
        # [TU remaining, HP remaining, unit accuracy, turn percentage completed, snap hit chance, aimed hit chance, snap TU cost, aimed TU cost , straight step TU cost, diagnol step TU cost ]
        # it is in percentages because its going to be easier for the network to learn
       
        # "grid" is an array that contains the 5 layers of the map [0-4]
        # [walls, the current active soldier, all friendly soldiers, visible aliens, health fraction of all units shown]
        # (Each layer is 12 by 12 and it is filled with 1s and 0s if its condition is true)
        # (Ex: the first layer as a 1 for all wall locations and a 0 for all non-wall locations)

        # stop while still having enough units for reaction snap_shot
        def end_turn(snap_cost,time_units,action_cost):
            # both arguments are in percentage
            # if desired action won't leave enough units for snap_shot then return true to pass turn
            return time_units - action_cost < snap_cost

        # if there is an alien, take snap shot (lowest number in the 8-11 range)
        # if you see an alien, try to kill it with snapshot
        if obs["grid"][3].any():
            # checking if it is legal to shoot the alien
            # and finding action needed to shoot alien (if possible)
            # Check which are legal and then pick lowest; this gives the snapshot action
            shots = {}
            for i in range(len(mask)):
                if mask[i] and i > 7 and i < len(mask)-1:
                    shots[i] = mask[i]
            if shots:
                return int(np.min([int(key) for key in shots.keys()]))
            return len(mask)-1

        # else move to opposite corner (bottom right since soldiers alway start top left)
        # aliens always start in bottom right so this makes them find the aliens
        # (might change this later when upgrading environment to be slowly less simple)

        # before moving we check if we move or end turn
        # (I'm being lazy and assuming we are making a diagnol step rather than checking if we are making straight step or diagnol step)
        if end_turn(obs["unit"][6],obs["unit"][0],obs["unit"][9]):
            # Note: ending turn is always the last option in the mask
            return len(mask)-1 
        if fixed_position: #(for now this is always true)
            # we always move toward bottom right corner?
            # checking if we can move by checking if there is a wall or soldier there (alien already taken care of)
            # directions we are checking [SE,S,E,SW,NE,W,N,NW]
            directions = [3,4,2,5,1,6,0,7]
            for direc in directions:
                if mask[direc]:
                    return direc
        # if we can't do anything, end turn
        return len(mask)-1

    def cautious_policy(obs,mask):
        # soldier holds still and waits to see an alien rather than chasing after them
        if obs["grid"][3].any():
            shots = {}
            for i in range(len(mask)):
                if mask[i] and i > 7 and i < len(mask)-1:
                    shots[i] = mask[i]
            if shots:
                return int(np.min([int(key) for key in shots.keys()]))
            return len(mask)-1
        return len(mask)-1


    return aggressive_policy, cautious_policy

def learning_agent(checkpoint):
    config = checkpoint["config"]
    policy_network = PolicyNetwork_Xcom(config["obs_size"], config["action_size"], config["hidden_size"])
    policy_network.load_state_dict(checkpoint["policy"])
    policy_network.eval() 

    def learning_policy(obs,mask):
        actions, _ = get_action_ppo_xcom(policy_network,obs[None,:],mask[None,:],is_training=False)
        return int(actions[0])

    # These should go in as **envkwargs because they don't fit anythign else in the arguments
    return learning_policy

def epoch_of(path):
    """The epoch number inside a checkpoint's file name, e.g. 200 from 'checkpoint_xcom_epoch200.pt'."""
    return int(re.search(r"epoch(\d+)", os.path.basename(path)).group(1))

def epoch_checkpoints(name=None):
    """Paths of one run's epoch checkpoints, in training order."""
    pattern = f"runs/checkpoint_xcom_epoch*_{name}.pt" if name else "runs/checkpoint_xcom_epoch*.pt"
    expected = rf"checkpoint_xcom_epoch\d+_{re.escape(name)}\.pt" if name else r"checkpoint_xcom_epoch\d+\.pt"
    paths = [p for p in glob.glob(pattern) if re.fullmatch(expected, os.path.basename(p))]
    if not paths: print("No checkpoints found that match name")
    return sorted(paths, key=epoch_of)

def get_best_policy(name=None,n_battles=300):
    # name = "final" then it checks for that appended onto end of everything
    # None is default, which means it look for final and no appended on epochs
    
    agents = []
    checkpoints = []
    # getting all epoch saves
    for path in epoch_checkpoints(name): 
        checkpoint = load_checkpoint(path)
        checkpoints.append((path,checkpoint))
        agents.append(learning_agent(checkpoint))
    # I can just get one since all the epochs work off of the same env_kwargs
    env_kwargs = checkpoint["config"]["env_kwargs"]


    # getting evaluation results for each
    results = []
    for (path, _), policy in tqdm(list(zip(checkpoints, agents)), desc="checkpoints"):
        r = evaluate(policy, flatten=True, n_battles=n_battles, init_seed=2_000_000,
                     desc=os.path.basename(path), **env_kwargs)
        results.append(r)
        tqdm.write(f"{os.path.basename(path):40s} win {r['win_rate']:6.1%} ± {r['ci95']:4.1%}")

    # checking them against each other to find best
    win_rates = [r["win_rate"] for r in results]
    index = int(np.argmax(win_rates))
    # returning best checkpoint
    return (checkpoints[index][0], checkpoints[index][1])
    
def evaluate_learner(name=None, n_battles=500):
    # getting environment variables
    check_path, checkpoint = get_best_policy(name)
    print(f"Checkpoint path for learner is: {check_path}")
    env_kwargs = checkpoint["config"]["env_kwargs"]

    # setting up all agents and their related vars
    aggressive, cautious = scripted_agent()
    agents = {                     
      "random": (random_agent(), True), # Note: True or False refers to need to flatten
      "scripted (aggressive)": (aggressive, False),   
      "scripted (cautious)": (cautious, False),
      "learner": (learning_agent(checkpoint), True),  
    }
    # evaluating agents
    results = {agent_name: evaluate(policy, n_battles=n_battles, flatten=flatten, desc=agent_name, **env_kwargs)
            for agent_name, (policy, flatten) in agents.items()}
    # printing evaluation
    for agent_name, r in results.items():
            print(f"{agent_name:22s} win {r['win_rate']:6.1%} ± {r['ci95']:4.1%} loss {r['loss_rate']:4.0%} timeout {r['timeout_rate']:4.0%}")

    # plotting agents results
    reference_lines = {agent_name: r["win_rate"] for agent_name, r in results.items() if agent_name != "learner"}
    reference_lines["learner (held-out)"] = results["learner"]["win_rate"]

    pic_path = f"runs/training_curves_{name}.png" if name else f"runs/training_curves_final.png"
    plot_history(check_path, pic_path, baselines=reference_lines)
    # returning them (in case I want to keep them)
    return results

if __name__ == "__main__":
    evaluate_learner()
