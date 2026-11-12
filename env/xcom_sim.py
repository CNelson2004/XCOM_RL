"""Minimal X-COM-like battle simulator (environment side only, no learning code)."""
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces

FLOOR, WALL = 0, 1
SOLDIER, ALIEN = 0, 1
DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # (dy, dx), N first, clockwise
SIGHT = 20                       # tiles, daytime
MOVE_TU = (4, 6)                 # orthogonal, diagonal (+50%)


@dataclass(frozen=True)
class Weapon:
    name: str
    power: int
    snap: tuple                  # (TU cost as fraction of max TU, accuracy multiplier)
    aimed: tuple

RIFLE = Weapon("rifle", 30, (0.25, 0.60), (0.80, 1.10))
PLASMA_PISTOL = Weapon("plasma pistol", 52, (0.30, 0.65), (0.60, 0.85))


@dataclass
class Unit:
    side: int
    y: int
    x: int
    max_tu: int
    max_hp: int
    reactions: int
    accuracy: int
    armor: int
    weapon: Weapon
    tu: int = 0
    hp: int = 0

    def __post_init__(self):
        self.tu, self.hp = self.max_tu, self.max_hp

    @property
    def alive(self):
        return self.hp > 0

    def shot_cost(self, mode):
        return int(self.max_tu * getattr(self.weapon, mode)[0])

    def initiative(self):        # reaction score: Reactions x fraction of TUs left
        return self.reactions * self.tu / self.max_tu


def line(y0, x0, y1, x1):
    """Bresenham: tiles from (y0, x0) to (y1, x1), both ends included."""
    dy, dx = abs(y1 - y0), abs(x1 - x0)
    sy, sx = (1 if y1 > y0 else -1), (1 if x1 > x0 else -1)
    err, pts = dx - dy, []
    while True:
        pts.append((y0, x0))
        if (y0, x0) == (y1, x1):
            return pts
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy


class Battle:
    """Rules of the game. Knows nothing about Gymnasium or neural networks."""

    def __init__(self, size=12, n_soldiers=2, n_aliens=2, max_turns=30, seed=None):
        self.size, self.n_soldiers, self.n_aliens, self.max_turns = size, n_soldiers, n_aliens, max_turns
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---------- setup ----------
    def reset(self):
        while True:              # regenerate until every floor tile is reachable
            g = np.zeros((self.size, self.size), dtype=np.int8)
            g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = WALL
            for _ in range(self.size // 2):
                y, x = self.rng.integers(2, self.size - 2, size=2)
                n = int(self.rng.integers(2, 5))
                if self.rng.random() < 0.5:
                    g[y, x:x + n] = WALL
                else:
                    g[y:y + n, x] = WALL
            g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = WALL
            if self._connected(g):
                break
        self.grid = g
        self.units = []
        r = self.rng
        for i in range(self.n_soldiers):   # rookies start in the top-left corner
            self._spawn(Unit(SOLDIER, 0, 0, int(r.integers(50, 61)), int(r.integers(25, 41)),
                             int(r.integers(30, 61)), int(r.integers(40, 71)), 0, RIFLE), near=(1, 1))
        for i in range(self.n_aliens):     # Sectoid soldiers start in the bottom-right corner
            self._spawn(Unit(ALIEN, 0, 0, 54, 30, 63, 52, 4, PLASMA_PISTOL), near=(self.size - 2, self.size - 2))
        self.turn, self.active, self.events = 1, 0, []
        self.last_seen = None
        self._update_last_seen()

    def _connected(self, g):
        free = list(zip(*np.where(g == FLOOR)))
        seen, stack = {free[0]}, [free[0]]
        while stack:
            y, x = stack.pop()
            for dy, dx in DIRS[::2]:
                p = (y + dy, x + dx)
                if g[p] == FLOOR and p not in seen:
                    seen.add(p); stack.append(p)
        return len(seen) == len(free)

    def _spawn(self, u, near):
        free = [p for p in zip(*np.where(self.grid == FLOOR)) if self.unit_at(*p) is None]
        u.y, u.x = min(free, key=lambda p: abs(p[0] - near[0]) + abs(p[1] - near[1]))
        self.units.append(u)

    # ---------- queries ----------
    def side(self, s):
        return [u for u in self.units if u.side == s and u.alive]

    def unit_at(self, y, x):
        return next((u for u in self.units if u.alive and (u.y, u.x) == (y, x)), None)

    def sees(self, a, b):
        if max(abs(a.y - b.y), abs(a.x - b.x)) > SIGHT:
            return False
        return all(self.grid[p] == FLOOR for p in line(a.y, a.x, b.y, b.x)[1:-1])

    def visible_enemies(self, side):
        """Squad-shared vision: enemies seen by at least one living unit of `side`."""
        mine, theirs = self.side(side), self.side(1 - side)
        return [e for e in theirs if any(self.sees(m, e) for m in mine)]

    def can_step(self, u, d):
        dy, dx = DIRS[d]
        y, x = u.y + dy, u.x + dx
        if self.grid[y, x] == WALL or self.unit_at(y, x) or u.tu < MOVE_TU[d % 2]:
            return False
        # no cutting corners diagonally
        return d % 2 == 0 or (self.grid[u.y + dy, u.x] == FLOOR and self.grid[u.y, u.x + dx] == FLOOR)

    def can_shoot(self, u, target, mode):
        return target.alive and u.tu >= u.shot_cost(mode) and self.sees(u, target)

    # ---------- actions ----------
    def step_unit(self, u, d):
        assert self.can_step(u, d)
        u.y, u.x, u.tu = u.y + DIRS[d][0], u.x + DIRS[d][1], u.tu - MOVE_TU[d % 2]
        self._update_last_seen()
        self._reaction_fire(u)

    def shoot(self, u, target, mode, reaction=False):
        assert self.can_shoot(u, target, mode)
        u.tu -= u.shot_cost(mode)
        p_hit = min(0.95, u.accuracy / 100 * getattr(u.weapon, mode)[1])
        dmg = 0
        if self.rng.random() < p_hit:
            dmg = max(0, int(self.rng.integers(0, 2 * u.weapon.power + 1)) - target.armor)   # 0-200% of power
            target.hp = max(0, target.hp - dmg)
        self.events.append(dict(type="shot", shooter=u.side, mode=mode, reaction=reaction,
                                hit=dmg > 0, damage=dmg, killed=not target.alive))
        if not reaction and u.alive:
            self._reaction_fire(u)

    def _reaction_fire(self, mover):
        """Enemies that see the mover and beat its initiative take a snap shot."""
        for e in self.side(1 - mover.side):
            if mover.alive and self.can_shoot(e, mover, "snap") and e.initiative() > mover.initiative():
                self.shoot(e, mover, "snap", reaction=True)

    def _update_last_seen(self):
        seen = self.visible_enemies(ALIEN)
        if seen:
            self.last_seen = (seen[0].y, seen[0].x)

    # ---------- turn structure ----------
    def end_unit_turn(self):
        """Pass control to the next living soldier; after the last one the aliens move."""
        self.active += 1
        while self.active < self.n_soldiers and not self.units[self.active].alive:
            self.active += 1
        if self.active >= self.n_soldiers:
            self._alien_turn()
            self.turn += 1
            for u in self.units:
                u.tu = u.max_tu
            self.active = next((i for i in range(self.n_soldiers) if self.units[i].alive), 0)

    def _alien_turn(self):
        for a in self.side(ALIEN):
            a.tu = a.max_tu
        for a in self.side(ALIEN):
            budget = a.max_tu // 2                      # keep half the TUs for reaction fire
            while a.alive and self.side(SOLDIER):
                targets = [s for s in self.side(SOLDIER) if self.can_shoot(a, s, "snap")]
                if targets:
                    self.shoot(a, min(targets, key=lambda s: abs(s.y - a.y) + abs(s.x - a.x)), "snap")
                    continue
                steps = [d for d in range(8) if self.can_step(a, d)]
                if not steps or a.tu <= budget:
                    break
                if self.last_seen:                      # walk toward the last place a soldier was seen
                    ty, tx = self.last_seen
                    best = min(max(abs(a.y + DIRS[d][0] - ty), abs(a.x + DIRS[d][1] - tx)) for d in steps)
                    steps = [d for d in steps if max(abs(a.y + DIRS[d][0] - ty), abs(a.x + DIRS[d][1] - tx)) == best]
                self.step_unit(a, int(self.rng.choice(steps)))

    def result(self):
        if not self.side(ALIEN):
            return "win"
        if not self.side(SOLDIER):
            return "loss"
        return "timeout" if self.turn > self.max_turns else None

    def render(self):
        rows = [["#" if c == WALL else "." for c in row] for row in self.grid]
        shown = self.side(SOLDIER) + self.visible_enemies(SOLDIER)
        for u in shown:
            rows[u.y][u.x] = "a" if u.side == ALIEN else str(self.units.index(u))
        return "\n".join("".join(r) for r in rows) + f"\nturn {self.turn}  active soldier {self.active}"


class XcomEnv(gym.Env):
    """Gymnasium wrapper. One step = one action by the currently active soldier.

    Actions: 0-7 step in direction d | 8..8+A-1 snap shot at alien i |
             8+A..8+2A-1 aimed shot at alien i | last = end this soldier's turn.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size=12, n_soldiers=2, n_aliens=2, max_turns=30, reward_fn=None, render_mode=None):
        self.cfg = dict(size=size, n_soldiers=n_soldiers, n_aliens=n_aliens, max_turns=max_turns)
        self.A, self.render_mode = n_aliens, render_mode
        self.reward_fn = reward_fn or sparse_reward
        self.action_space = spaces.Discrete(8 + 2 * n_aliens + 1)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 1, (5, size, size), np.float32),     # walls, active, friends, aliens, hp
            "unit": spaces.Box(0, 1, (4,), np.float32),                # TU, HP, accuracy, turn (all scaled)
        })
        self.battle = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.battle = Battle(**self.cfg, seed=int(self.np_random.integers(2**31)))
        return self._obs(), {"action_mask": self.action_masks()}

    def _aliens(self):
        return self.battle.units[self.cfg["n_soldiers"]:]

    def action_masks(self):
        b, u = self.battle, self.battle.units[self.battle.active]
        seen = b.visible_enemies(SOLDIER)
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[:8] = [b.can_step(u, d) for d in range(8)]
        for i, a in enumerate(self._aliens()):
            ok = a in seen                                             # may only target aliens the squad sees
            mask[8 + i] = ok and b.can_shoot(u, a, "snap")
            mask[8 + self.A + i] = ok and b.can_shoot(u, a, "aimed")
        mask[-1] = True                                                # ending the turn is always legal
        return mask

    def step(self, action):
        b = self.battle
        illegal = not self.action_masks()[action]
        if illegal:                                                    # masked-out action: treat as "end turn"
            action = self.action_space.n - 1
        u, b.events = b.units[b.active], []
        if action < 8:
            b.step_unit(u, action)
        elif action < 8 + 2 * self.A:
            i, mode = (action - 8) % self.A, ("snap" if action < 8 + self.A else "aimed")
            b.shoot(u, self._aliens()[i], mode)
        if action == self.action_space.n - 1 or not u.alive:
            b.end_unit_turn()
        res = b.result()
        if res is None and not b.units[b.active].alive:               # active soldier died on the alien turn
            b.end_unit_turn(); res = b.result()
        terminated, truncated = res in ("win", "loss"), res == "timeout"
        info = {"events": b.events, "result": res, "illegal": illegal, "action_mask": self.action_masks(),
                "soldiers_alive": len(b.side(SOLDIER)), "turn": b.turn}
        return self._obs(), self.reward_fn(b.events, res), terminated, truncated, info

    def _obs(self):
        b, u = self.battle, self.battle.units[self.battle.active]
        g = np.zeros(self.observation_space["grid"].shape, np.float32)
        g[0] = b.grid == WALL
        g[1, u.y, u.x] = 1
        for s in b.side(SOLDIER):
            g[2, s.y, s.x] = 1
            g[4, s.y, s.x] = s.hp / s.max_hp
        for a in b.visible_enemies(SOLDIER):
            g[3, a.y, a.x] = 1
            g[4, a.y, a.x] = a.hp / a.max_hp
        vec = np.array([u.tu / u.max_tu, u.hp / u.max_hp, u.accuracy / 100, b.turn / b.max_turns], np.float32)
        return {"grid": g, "unit": np.clip(vec, 0, 1)}

    def render(self):
        return self.battle.render()


def sparse_reward(events, result):
    """Placeholder: +1 win, -1 loss, 0 otherwise. Reward shaping is yours to design (see Phase 3)."""
    return {"win": 1.0, "loss": -1.0}.get(result, 0.0)
