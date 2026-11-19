
def tu_reward(events, result):
    # gives a small reward for fraction of time units left at the end of the turn
    # (this encourages keeping time units for reaction fire)
    # will deal with balancing this with exploration later
    # Note: results = "win" or "loss" or "timeout" or None
    # events is a list of dictionaries of everything that happened during this step
    # type tells you the specific event and it lets you know the rest of the keys

    rewards = 0.0
    # rewards for winning or not
    if result == "win":
        rewards += 1.0
    elif result == "loss":
        rewards += -1.0
    elif result == "timeout":
        rewards += 0.0

    # rewards for TU saved
    # Note: event is a dictionary, events is a list
    for event in events:
        if event["type"] == "end_turn" and event["aliens_in_view"] == 0:
            # giving reward for holding TU for reaction fire
            tu_left = event["tu_left"]
            tu_max = event["tu_max"]
            rewards += (tu_left/tu_max) * 0.005

    return rewards

def tu_and_damage_reward(events, result):
    # rewards soldier for holding TU for reaction fire
    # also balances it with giving reward for dealing damage to alien

    rewards = 0.0
    # rewards for winning or not
    if result == "win":
        rewards += 1.0
    elif result == "loss":
        rewards += -1.0
    elif result == "timeout":
        rewards += 0.0

    for event in events:
        # rewards for damage
        if event["type"] == "shot" and event["hit"]:
            # rewards for damage dealt to aliens
            frac_health = event["hp_lost"]/event["target_max_hp"]
            if event["shooter"] == 0: # shooter is soldier
                # we don't have to check damage because we know we hit
                rewards += frac_health * 0.1
                if event["killed"]:
                    rewards += 0.1
            # penalty for damage taken
            else: # shooter == 1 (alien)
                rewards += frac_health * -0.1
                if event["killed"]:
                    rewards += -0.1

        # rewards for TU saved
        if event["type"] == "end_turn" and event["aliens_in_view"] == 0:
            # giving reward for holding TU for reaction fire
            tu_left = event["tu_left"]
            tu_max = event["tu_max"]
            rewards += (tu_left/tu_max) * 0.005

    return rewards

    # Note: could think about giving very small reward for hp remaining on a soldier
    # (Would only do this if model becomes aggressive and TU reward isn't enough to handle it.)
