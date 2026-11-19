"""Rule tests for the simulator. Run with: pytest test_xcom_sim.py"""
import numpy as np
from env.xcom_sim import Battle, XcomEnv, line, format_event, SOLDIER, ALIEN, WALL, FLOOR, UNIT_FEATURES


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
    assert obs["unit"].shape == (10,) and env.observation_space["unit"].contains(obs["unit"])
    assert np.isclose(f["snap_hit_chance"], b.hit_chance(u, "snap"))
    assert np.isclose(f["aimed_hit_chance"], b.hit_chance(u, "aimed"))
    assert np.isclose(f["snap_tu_cost"], u.shot_cost("snap") / u.max_tu)
    assert np.isclose(f["aimed_tu_cost"], u.shot_cost("aimed") / u.max_tu)
    assert f"({f['snap_hit_chance']:.0%} chance)" in env.describe_action(8, with_chance=True)


def test_step_costs_in_observation_match_real_steps():
    """A straight step lowers tu_fraction by step_tu_cost, a diagonal one by diagonal_step_tu_cost."""
    env = XcomEnv()
    obs, info = env.reset(seed=0)
    b = env.battle
    b.grid[1:-1, 1:-1] = FLOOR                       # open map, so both kinds of step are legal
    b.units[2].tu = b.units[3].tu = 0                # no reaction fire
    obs = env._obs()
    f = dict(zip(UNIT_FEATURES, obs["unit"]))
    assert np.isclose(f["diagonal_step_tu_cost"], 1.5 * f["step_tu_cost"])
    for action, key in [(4, "step_tu_cost"), (3, "diagonal_step_tu_cost")]:   # step S, then step SE
        before = dict(zip(UNIT_FEATURES, env._obs()["unit"]))["tu_fraction"]
        obs, _, _, _, info = env.step(action)
        after = dict(zip(UNIT_FEATURES, obs["unit"]))["tu_fraction"]
        assert np.isclose(before - after, f[key])


def test_alien_hp_setting():
    env = XcomEnv(n_soldiers=1, n_aliens=1, alien_hp=1)
    env.reset(seed=0)
    alien = env.battle.units[1]
    assert alien.side == ALIEN and alien.hp == alien.max_hp == 1
    default = XcomEnv()
    default.reset(seed=0)
    assert default.battle.units[2].max_hp == 30       # default unchanged


def test_any_hit_kills_a_1_hp_alien():
    b = Battle(size=6, n_soldiers=1, n_aliens=1, seed=0, alien_hp=1)
    s, a = b.units
    a.tu = 0                                          # no reaction fire
    for _ in range(50):                               # keep shooting until a hit lands
        s.tu = s.max_tu
        b.shoot(s, a, "aimed")
        if b.events[-1]["hit"] or not a.alive:
            break
    assert not a.alive


def test_small_open_maps_start_in_view():
    """On 5 x 5 and 6 x 6 maps there are no interior walls, so the alien is always visible at the start."""
    for size in (5, 6):
        env = XcomEnv(size=size, n_soldiers=1, n_aliens=1)
        for k in range(50):
            obs, _ = env.reset(seed=k)
            assert obs["grid"].shape == (5, size, size)
            assert obs["grid"][3].any()
            assert not obs["grid"][0][1:-1, 1:-1].any()   # no interior walls


def test_wall_segments_setting_and_size_limits():
    b = Battle(size=12, seed=0, wall_segments=0)
    assert not b.grid[1:-1, 1:-1].any()
    assert Battle(size=12, seed=0).wall_segments == 6     # default 12 x 12 map keeps its 6 wall pieces
    for bad in (dict(size=4), dict(size=5, n_soldiers=5, n_aliens=5), dict(alien_hp=0)):
        try:
            Battle(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_vector_info_always_has_illegal_and_result():
    """Even when every copy finishes on the same step, the merged info keeps both keys.
    The finished battles' real values stay in final_info."""
    from gymnasium.wrappers import FlattenObservation
    from gymnasium.vector import SyncVectorEnv, AutoresetMode
    envs = SyncVectorEnv([lambda: FlattenObservation(XcomEnv(size=5, n_soldiers=1, n_aliens=1, alien_hp=1))
                          for _ in range(6)], autoreset_mode=AutoresetMode.SAME_STEP)
    _, info = envs.reset(seed=0)
    assert "illegal" in info and "result" in info
    rng, all_done_steps = np.random.default_rng(0), 0
    for _ in range(3000):
        acts = np.array([rng.choice(np.flatnonzero(m)) for m in info["action_mask"]])
        _, _, te, tr, info = envs.step(acts)
        done = te | tr
        assert "illegal" in info and "result" in info
        if done.all():
            all_done_steps += 1
        for i in np.flatnonzero(done):
            assert not info["illegal"][i] and info["result"][i] is None      # new battle's placeholders
            assert info["final_info"]["result"][i] in ("win", "loss", "timeout")
    assert all_done_steps > 0          # the situation that used to drop the keys did occur


def test_end_turn_event_reports_tus_kept():
    """Ending a turn records the soldier's TUs left, before the alien turn begins."""
    env = XcomEnv()
    env.reset(seed=0)
    b = env.battle
    b.grid[1:-1, 1:-1] = FLOOR                       # open map, so a step is legal
    b.units[2].tu = b.units[3].tu = 0                # no reaction fire
    u = b.units[b.active]
    env.step(4)                                      # one step south costs 4 TU
    _, _, _, _, info = env.step(env.action_space.n - 1)
    ends = [e for e in info["events"] if e["type"] == "end_turn"]
    assert len(ends) == 1
    e = ends[0]
    assert e["unit"] == "soldier 0" and e["tu_max"] == u.max_tu and e["tu_left"] == u.max_tu - 4
    assert e["aliens_in_view"] == len(b.visible_enemies(SOLDIER))   # nobody moved since the turn ended
    assert format_event(e).startswith(f"soldier 0 ends turn with {u.max_tu - 4}/{u.max_tu} TU")


def test_end_turn_event_comes_before_alien_turn_and_only_on_end_turn():
    env = XcomEnv()
    env.reset(seed=0)
    last = env.action_space.n - 1
    _, _, _, _, info = env.step(last)                # soldier 0 ends: no alien turn yet
    assert [e["type"] for e in info["events"]] == ["end_turn"]
    _, _, _, _, info = env.step(last)                # soldier 1 ends: the aliens move after the record
    types = [e["type"] for e in info["events"]]
    assert types[0] == "end_turn" and "alien_turn" in types
    assert types.index("end_turn") < types.index("alien_turn")
    obs, info = env.reset(seed=0)
    for a in range(8):                               # a move never produces an end_turn record
        if info["action_mask"][a]:
            _, _, _, _, info = env.step(a)
            assert all(e["type"] != "end_turn" for e in info["events"])
            break


def test_shot_event_reports_hp_lost_capped_and_max_hp():
    """hp_lost is the HP the target actually lost (never more than it had); target_max_hp is its full HP."""
    b, s, a = open_battle()
    b.units[3].tu = 0; a.tu = 0                     # no reaction fire back
    a.hp = 1                                         # any hit overkills
    for _ in range(50):                              # keep shooting until a hit lands
        s.tu = s.max_tu
        b.events = []
        b.shoot(s, a, "snap")
        e = b.events[0]
        assert e["target_max_hp"] == a.max_hp
        if e["hit"]:
            assert e["hp_lost"] == 1 and e["damage"] >= 1 and e["killed"]
            break
        assert e["hp_lost"] == 0
    else:
        raise AssertionError("no hit in 50 shots")


def test_units_with_identical_stats_keep_their_own_names():
    """A dead alien and a living one on the same tile must not be mixed up (units compare by identity)."""
    b = Battle(size=12, n_soldiers=2, n_aliens=2, seed=0)
    a, c = b.units[2], b.units[3]
    a.hp = c.hp = 0
    a.y, a.x, c.y, c.x = 5, 5, 5, 5
    a.tu = c.tu = 0                                  # every field now equal
    assert b.name(a) == "alien a" and b.name(c) == "alien b"
