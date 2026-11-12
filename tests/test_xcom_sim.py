"""Rule tests for the simulator. Run with: pytest test_xcom_sim.py"""
import numpy as np
from xcom_sim import Battle, XcomEnv, line, SOLDIER, ALIEN, WALL, FLOOR


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
    assert not any(e["reaction"] for e in b.events)
    s.reactions, a.reactions = 10, 100              # now the alien reacts
    b.step_unit(s, 2)
    assert any(e["reaction"] for e in b.events)


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
