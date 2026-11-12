"""Rule tests for the simulator. Run with: pytest test_xcom_sim.py"""
import numpy as np
from env.xcom_sim import Battle, XcomEnv, line, SOLDIER, ALIEN, WALL, FLOOR, UNIT_FEATURES


def open_battle():
    b = Battle(size=12, seed=0)
    b.grid[1:-1, 1:-1] = FLOOR                      # clear interior walls for predictable tests
    s, a = b.units[0], b.units[2]
    s.y, s.x, a.y, a.x = 5, 2, 5, 8
    b.units[1].y, b.units[1].x, b.units[3].y, b.units[3].x = 1, 1, 10, 10
    return b, s, a


def test_line_endpoints_and_length():
    pts = line(0, 0, 3, 7)
    assert pts[0] == (0, 0) and pts[-1] == (3, 7) and len(pts) == 8


def test_move_costs():
    b, s, _ = open_battle()
    b.units[2].tu = 0; b.units[3].tu = 0            # no reaction fire
    tu = s.tu
    b.step_unit(s, 2); assert s.tu == tu - 4        # east, orthogonal
    b.step_unit(s, 1); assert s.tu == tu - 10       # north-east, diagonal


def test_wall_blocks_sight():
    b, s, a = open_battle()
    assert b.sees(s, a)
    b.grid[5, 5] = WALL
    assert not b.sees(s, a) and not b.can_shoot(s, a, "snap")


def test_shot_costs_percent_of_max_tu():
    b, s, a = open_battle()
    a.tu = 0
    tu = s.tu
    b.shoot(s, a, "snap")
    assert s.tu == tu - int(s.max_tu * 0.25)


def test_reaction_needs_higher_initiative():
    b, s, a = open_battle()
    b.units[3].tu = 0
    s.reactions, a.reactions = 100, 10              # soldier keeps the initiative
    b.step_unit(s, 2)
    assert not any(e["type"] == "shot" and e["reaction"] for e in b.events)
    s.reactions, a.reactions = 10, 100              # now the alien reacts
    b.step_unit(s, 2)
    assert any(e["type"] == "shot" and e["reaction"] for e in b.events)


def test_mask_never_allows_illegal_and_same_seed_same_game():
    def play(seed):
        env, rng, log = XcomEnv(), np.random.default_rng(1), []
        _, info = env.reset(seed=seed)
        for _ in range(300):
            a = rng.choice(np.flatnonzero(info["action_mask"]))
            _, r, term, trunc, info = env.step(a)
            assert not info["illegal"]
            log.append((int(a), r, info["turn"]))
            if term or trunc:
                break
        return log
    assert play(5) == play(5)


def test_sight_is_symmetric():
    for seed in range(30):
        b = Battle(size=12, seed=seed)
        for u in b.units:
            for v in b.units:
                assert b.sees(u, v) == b.sees(v, u)


def test_events_are_readable():
    b, s, a = open_battle()
    b.units[2].tu = b.units[3].tu = 0                # no reaction fire
    b.shoot(s, a, "aimed")
    e = b.events[0]
    assert e["shooter_name"] == "soldier 0" and e["target_name"] == "alien a"
    assert e["target_hp"] == a.hp and 0 < e["hit_chance"] <= 0.95


def test_spotting_events():
    b, s, a = open_battle()
    b.grid[1:7, 5] = WALL                            # wall column hides alien a from the soldier
    s.y, s.x, a.y, a.x = 6, 4, 7, 8
    b.in_view = b._visible_ids()
    b.units[2].tu = b.units[3].tu = 0                # no reaction fire
    assert not b.sees(s, a)
    b.step_unit(s, 4)                                # step south, out from behind the wall
    assert b.sees(s, a)
    assert any(e["type"] == "spotted" and e["unit"] == "alien a" for e in b.events)
    b.step_unit(s, 0)                                # step back north
    assert any(e["type"] == "lost_sight" and e["unit"] == "alien a" for e in b.events)


def test_describe_action():
    env = XcomEnv()
    assert env.describe_action(0) == "step N"
    assert env.describe_action(8) == "snap shot at alien a"
    assert env.describe_action(9) == "aimed shot at alien a"
    assert env.describe_action(10) == "snap shot at alien b"
    assert env.describe_action(11) == "aimed shot at alien b"
    assert env.describe_action(12) == "end turn"


def test_shot_actions_hit_the_named_alien():
    """Each shot action fires the named shot type at the named alien."""
    for action, (alien, mode) in {8: (0, "snap"), 9: (0, "aimed"), 10: (1, "snap"), 11: (1, "aimed")}.items():
        env = XcomEnv()
        env.reset(seed=0)
        b = env.battle
        b.grid[1:-1, 1:-1] = FLOOR
        s = b.units[0]
        for k, (y, x) in enumerate([(5, 2), (1, 1), (5, 8), (9, 8)]):
            b.units[k].y, b.units[k].x = y, x
        b.units[2].tu = b.units[3].tu = 0            # no reaction fire
        target = b.units[2 + alien]
        assert env.action_masks()[action]
        tu = s.tu
        _, _, _, _, info = env.step(action)
        shot = [e for e in info["events"] if e["type"] == "shot"][0]
        assert shot["target_name"] == f"alien {'ab'[alien]}" and shot["mode"] == mode
        assert s.tu == tu - s.shot_cost(mode)


def test_chance_shown_before_matches_chance_used():
    env = XcomEnv()
    env.reset(seed=0)
    b = env.battle
    u = b.units[b.active]
    text = env.describe_action(9, with_chance=True)
    assert text == f"aimed shot at alien a ({b.hit_chance(u, 'aimed'):.0%} chance)"


def test_unit_observation_shows_hit_chances_and_costs():
    """The agent sees the same hit chances a player sees, and the time-unit cost of each shot."""
    env = XcomEnv()
    obs, _ = env.reset(seed=0)
    b = env.battle
    u = b.units[b.active]
    f = dict(zip(UNIT_FEATURES, obs["unit"]))
    assert obs["unit"].shape == (8,) and env.observation_space["unit"].contains(obs["unit"])
    assert np.isclose(f["snap_hit_chance"], b.hit_chance(u, "snap"))
    assert np.isclose(f["aimed_hit_chance"], b.hit_chance(u, "aimed"))
    assert np.isclose(f["snap_tu_cost"], u.shot_cost("snap") / u.max_tu)
    assert np.isclose(f["aimed_tu_cost"], u.shot_cost("aimed") / u.max_tu)
    assert f"({f['snap_hit_chance']:.0%} chance)" in env.describe_action(8, with_chance=True)
