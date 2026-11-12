import numpy as np
from gymnasium.wrappers import FlattenObservation
from env.xcom_sim import XcomEnv

# evaluate the policy given to it
def evaluate(policy, n_battles=500,init_seed=1_000_000,flatten=True,**env_kwargs):
    # Note: env_kwargs includes: map_size, num_soldiers, num_aliens, max_turns, reward_function, render_mode
    # Defaults: 12,2,2,30,None,None
    # (map is 12 x 12 with 2 soldiers, 2 aliens and 30 max turns)

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
    for i in range(n_battles):
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
    # returning win, loss, and timeout percentage (aka rate), how many soldiers alive (on average) and turns taken (on average)
    return dict(win_rate=results.count("win")/n_battles, loss_rate=results.count("loss")/n_battles, timeout_rate=results.count("timeout")/n_battles,
                    soldiers_alive=float(np.mean(alive)), turns=float(np.mean(turns)))


# making agent that chooses actions at random so we have a baseline to beat
def random_agent():
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


if __name__ == "__main__":
    # creating baselines to compare against
    # using random agent for evaluation
    print("Random:")
    print(evaluate(random_agent()))
    # aggressive agent that chases aliens down
    print("Aggressive:")
    print(evaluate(scripted_agent()[0],flatten=False))
    # caustious agent that waits for aliens to come to them
    print("Cautious:")
    print(evaluate(scripted_agent()[1],flatten=False))

    # my agent
    print("Learner:")

