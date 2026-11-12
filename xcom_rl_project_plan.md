# Teaching a Neural Network to Fight X-COM Battles: Project Plan

**Goal:** train a neural network with reinforcement learning (RL) to win tactical battles in *X-COM: UFO Defense*, starting in a small Python simulator and moving toward the real game through OpenXcom Extended.

**How to read this plan**

- Phases 0 to 4 are the **core plan**. They fit your 30+ hands-on hours and are enough to show real progress.
- Phases 5 to 7, below the marked line, are **stretch goals**.
- **Code policy.** Environment sections contain code (AI-written code is allowed there). RL sections describe files, classes and methods in plain English only, with no executable code, because that code is yours to write.
- **Two kinds of time.** "Hands-on" is you at the keyboard. "Training" is the laptop running unattended and does not count against your 30 hours.
- Two files come with this plan: `xcom_sim.py` (a working starter simulator) and `test_xcom_sim.py` (its rule tests). Both run; the tests pass.

---

## 1. Corrections to the original plan

| Original plan | Correction | Why |
|---|---|---|
| "OpenXcom is the game's original code translated into C++" | OpenXcom is a from-scratch, open-source **reimplementation** of the engine in C++ (GPL v3). It still needs the original data files from your Steam copy. | The original game was a closed DOS program; nobody translated its code. This matters because OpenXcom's alien AI is a re-creation, close to the original but not identical. |
| Renderer is the last step (step 5) | Build a simple text renderer **first** | You cannot debug an agent you cannot watch. "It won't learn" is usually an environment bug that is obvious on screen. The starter simulator already has one. |
| Move to OpenXcom (step 3) before adding complexity (step 4) | Scale up **inside the Python simulator first**, then move | Simulator changes take minutes; engine changes take hours. Bigger squads and maps teach you the RL lessons without any C++. |
| Success = all aliens eliminated | Aliens **killed or stunned** counts as a win; add **turn limit** and **baselines** | Without a random and a scripted baseline, a 60% win rate means nothing. |
| No evaluation method | Fixed evaluation: 500 battles on held-out seeds, reported with a confidence interval | RL results are noisy; one good run proves little. |
| One agent per soldier (earlier idea) | **One shared network**, called once per soldier action, with a team reward | Already discussed; the simulator's turn structure is built around it. |

Things your plan did not include that this one adds: a setup phase, baselines, an evaluation protocol, an observation/action/reward specification, a debugging checklist, a risk list, and a glossary.

---

## 2. Scope and schedule

**Budget:** 30+ hands-on hours over 8 weeks; at least 20 of them on RL code, about 10 on the environment. Because the starter simulator is provided, the environment hours go to understanding and extending it instead of writing it from zero.

| Week | Phase | Hands-on (env) | Hands-on (RL) | Unattended training |
|---|---|---|---|---|
| 0 | Phase 0: your course lab (DQN + PPO on CartPole) | 0 | (lab time) | minutes |
| 1 | Phase 1: setup, read and extend the simulator | 4 h | 0 | 0 |
| 2 | Phase 2: baselines and evaluation | 2 h | 2 h | under 1 h |
| 3–4 | Phase 3: your own masked PPO on 2 vs 2 | 1 h | 10 h | 5–15 h total |
| 5–6 | Phase 4: scale up (curriculum) + library comparison | 3 h | 8 h | 20–60 h total |
| 7–8 | Write-up; optionally Phase 5 | 0–3 h | 2 h | leftover runs |
| | **Total** | **about 10–13 h** | **about 22 h** | **well inside 2 months** |

**Minimum viable success** (end of Phase 3): your own PPO beats the random baseline by a wide margin and at least matches the scripted baseline on 2 soldiers vs 2 Sectoids, with learning curves to show it. Everything after that is a bonus.

**Using training time well**

- Start a run when you stop working; read the curves when you come back. Never sit and watch a run.
- Save a checkpoint every 30 minutes and make runs resumable, so a sleeping laptop costs you nothing.
- Plug the laptop in, set Windows power mode to "Best performance", and disable sleep during runs. Keep it on a hard surface; thin laptops slow down when hot.
- Change one thing per run and write down what you changed before you start it.

**Training-time estimates are rough.** The starter simulator ran about 4,000 random-play steps per second on one core of my test machine; yours will differ. With a small network in the loop, expect a few hundred to a couple of thousand steps per second across 6 to 8 parallel environments. Measure it in Phase 2 and rescale the estimates.

**Colab:** your networks are small and the simulator runs on CPU, so a GPU helps little. Colab is useful mainly as a second machine for running a sweep in parallel. Your lab code calls `.cuda()`; on the laptop you will need to use the CPU device instead.

---

## 3. Architecture

```
            +--------------------------- Python ---------------------------+
            |                                                              |
            |   train.py  <--->  PPO agent (your code)                     |
            |                        |  action          ^  obs, reward,    |
            |                        v                  |  action mask     |
            |                 Gymnasium interface (XcomEnv)                |
            |                   /                         \                |
            |      Battle (Python simulator)        OxceEnv (stretch)      |
            |      Phases 1-4                         |  TCP, JSON lines   |
            +-----------------------------------------|--------------------+
                                                      v
                                     OpenXcom Extended + bridge patch (C++)
                                     Phases 5-6
```

The agent only ever sees the Gymnasium interface. If the simulator and the real-game bridge expose the **same observation and action format**, the training code does not change when you switch.

### Suggested folder layout

```
xcom_rl/
  env/
    xcom_sim.py          # Battle rules + XcomEnv (provided)
    rewards.py           # reward functions (yours: RL design)
    scripted.py          # scripted soldier policy for the baseline
    oxce_env.py          # stretch: client for the OpenXcom bridge
  agents/                # ALL yours
    networks.py
    buffer.py
    ppo.py
    dqn.py               # optional
  train.py               # yours
  evaluate.py            # evaluation harness (environment-side tooling)
  watch.py               # replay a saved battle in the terminal
  configs/               # one YAML file per experiment
  tests/test_xcom_sim.py # provided
  runs/                  # TensorBoard logs and checkpoints (git-ignored)
  notebooks/             # plots only
```

---

## 4. Environment specification (the contract everything shares)

This is what the provided `XcomEnv` implements. Treat it as version 1; change it deliberately and re-run baselines when you do.

**Decision unit.** One step is one action by the currently active soldier. Soldiers act in a fixed order; each acts until it chooses "end turn". After the last soldier, the aliens take their turn.

**Observation** (a dictionary):

| Key | Shape | Contents |
|---|---|---|
| `grid` | 5 × H × W | Channels: walls, active soldier, all friendly soldiers, **visible** aliens, health fraction of units shown |
| `unit` | 4 | Active soldier's TU fraction, health fraction, firing accuracy, and turn number (all scaled 0 to 1) |

Aliens appear only while at least one soldier has line of sight (squad-shared fog of war).

**Actions** (discrete, `8 + 2A + 1` where A is the number of aliens):

| Index | Action |
|---|---|
| 0–7 | Step one tile (N, NE, E, SE, S, SW, W, NW) |
| 8 … 8+A−1 | Snap shot at alien *i* |
| 8+A … 8+2A−1 | Aimed shot at alien *i* |
| last | End this soldier's turn |

**Action mask.** `env.action_masks()` and `info["action_mask"]` give a boolean array of the actions that are legal right now. An illegal action is treated as "end turn" and flagged in `info["illegal"]`, but your agent should never pick one.

**Reward.** The default is sparse: +1 win, −1 loss, 0 otherwise. `info["events"]` lists every shot (who fired, hit or miss, damage, kill), so you can build shaped rewards without touching the simulator. Reward design is an RL decision and is yours (Phase 3).

**Episode end.** Win (no aliens left), loss (no soldiers left), or truncation at the turn limit.

### Rules in the simulator and where they come from

| Rule | Value used | Source |
|---|---|---|
| Walking cost | 4 TU per tile, diagonals 50% more (6) | [Time Units](https://www.ufopaedia.org/Time_Units) |
| Shot TU cost | Percentage of the unit's **maximum** TUs | [Time Units](https://www.ufopaedia.org/Time_Units) |
| Rifle | Power 30; snap 25% TU / 60% accuracy; aimed 80% / 110% | [Rifle](https://www.ufopaedia.org/Rifle) |
| Plasma pistol | Power 52; snap 30% / 65%; aimed 60% / 85% | [Plasma Pistol](https://www.ufopaedia.org/Plasma_Pistol) |
| Chance to hit | Firing accuracy × weapon mode accuracy | [Firing Accuracy](https://www.ufopaedia.org/Firing_Accuracy) |
| Reaction fire | Score = Reactions × (current TU ÷ max TU); the watcher fires a snap shot if its score is higher | [Reaction fire](https://www.ufopaedia.org/Reaction_fire) |
| Sectoid soldier | 54 TU, 30 health, 63 reactions, 52 accuracy, 4 front armour | [Alien Stats](https://www.ufopaedia.org/Alien_Stats) |
| Rookie | TU 50–60, firing accuracy 40–70 | [Time Units](https://www.ufopaedia.org/Time_Units), [Firing Accuracy](https://www.ufopaedia.org/Firing_Accuracy) |
| Rookie health 25–40, reactions 30–60 | From memory, not re-checked | [Soldiers](https://www.ufopaedia.org/Soldiers) |
| Sight range 20 tiles (day) | From memory, not re-checked | [Line of sight](https://www.ufopaedia.org/Line_of_sight) |
| Damage 0–200% of weapon power, minus armour | From memory, not re-checked | [Damage](https://www.ufopaedia.org/Damage) |

**Known simplifications** (each is a candidate upgrade in Phase 4):

- A shot is one hit roll. The real game simulates a bullet path, so misses can hit other things and the displayed percentage is not the true hit chance.
- No facing: units see in all directions. In the real game units see a forward cone and turning costs 1 TU per 45°.
- No kneeling (+15% accuracy), auto shots, grenades, ammunition, morale, energy, or multiple floors.
- Aliens use a simple script: shoot if they can, otherwise walk toward the last place a soldier was seen, keeping half their TUs for reaction fire.
- "Mutual surprise" (no reaction fire in the instant both sides first see each other) is not modelled.
- On Beginner difficulty the real game halves alien accuracy and armour; the simulator uses the full values.

![Sectoid](https://www.ufopaedia.org/images/1/16/Sectoid.png)
*Your first opponent. Image: UFOpaedia.*

---

# CORE PLAN

## Phase 0: Course lab (DQN and PPO on CartPole)

**Goal.** Finish the lab so you have a PPO you understand line by line.

**What carries over, and what must change.** Your lab's structure maps onto this project as follows.

| Lab piece | Reuse | What changes for X-COM |
|---|---|---|
| `get_action_ppo` | Same idea: sample from the policy's distribution | Must apply the **action mask** before sampling, so illegal actions get zero probability |
| `PolicyNetwork` ending in a softmax | Same role | Input is a grid plus a vector, so it needs a small convolutional encoder. Output the raw scores and apply the mask before the softmax, otherwise masked actions keep probability |
| `ValueNetwork` | Same role | Can share the encoder with the policy |
| `calculate_return` (discounted sum to the end of the episode) | Works as a first version | Episodes are 40+ steps with late rewards, so the lab's γ = 0.9 forgets the outcome too fast; you will want a value near 0.99. An episode cut off by the turn limit is not the same as a loss; handle the two differently |
| `learn_ppo` (clipped loss, advantage = return − value, detached) | The core is unchanged | The mask used when the action was chosen must be stored and re-applied when recomputing probabilities, or the ratio is wrong. Add an entropy bonus and normalise advantages per batch |
| Memory tuples `(state, action, action_dist, return)` | Same idea | Add the mask. Storing the chosen action's log-probability is enough; you don't need the whole distribution |
| `ppo_main` collecting 100 episodes per epoch | Same loop shape | Collect a fixed number of **steps** from several parallel environments instead |
| `.cuda()` everywhere | No | CPU on the laptop; make the device a setting |
| DQN part | Optional comparison | The mask applies in two places: choosing the best action, and the "best next action" inside the target. The lab's replay list grows without limit; cap it |

**Before touching X-COM:** change your lab PPO to run on CPU with 4 parallel CartPole environments. If it still solves CartPole, your changes are sound.

**Resources.** [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) · [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) · [Invalid action masking paper](https://arxiv.org/abs/2006.14171)

---

## Phase 1: Setup and the simulator (environment)

**Goal.** A simulator you trust, because you have read it, tested it, and watched it.

**Hands-on:** 4 h. **Training:** none.

**Approach**

1. Create the project and a virtual environment.

   ```powershell
   mkdir xcom_rl; cd xcom_rl; git init
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install numpy gymnasium torch tensorboard pytest pyyaml
   ```

2. Drop in `xcom_sim.py` and `test_xcom_sim.py`, then run `pytest`. All six tests should pass.
3. Read `xcom_sim.py` top to bottom (about 300 lines). It has two classes:

   | Class | Job | Key methods |
   |---|---|---|
   | `Battle` | The rules. Knows nothing about RL. | `reset`, `sees`, `visible_enemies`, `can_step`, `step_unit`, `can_shoot`, `shoot`, `_reaction_fire`, `end_unit_turn`, `_alien_turn`, `result`, `render` |
   | `XcomEnv` | Gymnasium wrapper | `reset`, `step`, `action_masks`, `_obs`, `render` |

4. Play it by hand for ten minutes with the loop below. Nothing builds intuition for reward design faster.

   ```python
   # play.py: you are the policy
   from xcom_sim import XcomEnv
   import numpy as np
   env = XcomEnv(); obs, info = env.reset(seed=1); done = False
   while not done:
       print(env.render()); print("legal:", np.flatnonzero(info["action_mask"]))
       obs, r, term, trunc, info = env.step(int(input("action> ")))
       for e in info["events"]: print(e)
       done = term or trunc
   print(info["result"])
   ```

5. Add a test for any rule you change. Three pieces worth understanding in detail:

   ```python
   # Line of sight: walk a straight line of tiles; any wall strictly between blocks it.
   def sees(self, a, b):
       if max(abs(a.y - b.y), abs(a.x - b.x)) > SIGHT:
           return False
       return all(self.grid[p] == FLOOR for p in line(a.y, a.x, b.y, b.x)[1:-1])

   # Reaction fire: after a unit moves or shoots, each enemy that sees it and has
   # the higher initiative (Reactions x TU fraction) takes a snap shot.
   def _reaction_fire(self, mover):
       for e in self.side(1 - mover.side):
           if mover.alive and self.can_shoot(e, mover, "snap") and e.initiative() > mover.initiative():
               self.shoot(e, mover, "snap", reaction=True)

   # Action mask: what the active soldier may legally do right now.
   def action_masks(self):
       b, u = self.battle, self.battle.units[self.battle.active]
       seen = b.visible_enemies(SOLDIER)
       mask = np.zeros(self.action_space.n, dtype=bool)
       mask[:8] = [b.can_step(u, d) for d in range(8)]
       for i, a in enumerate(self._aliens()):
           ok = a in seen
           mask[8 + i] = ok and b.can_shoot(u, a, "snap")
           mask[8 + self.A + i] = ok and b.can_shoot(u, a, "aimed")
       mask[-1] = True
       return mask
   ```

   The reaction rule has a tactical consequence the agent must discover: a soldier who spends all its TUs walking has zero initiative and gets shot; one who stops early can shoot first.

**Measures of success**

- All tests pass, and the same seed replays the same battle.
- You can explain every rule in the table in section 4.
- You have won a battle by hand.

**Resources.** [Gymnasium: create a custom environment](https://gymnasium.farama.org/introduction/create_custom_env/) · UFOpaedia pages in section 4.

---

## Phase 2: Baselines and evaluation (mostly environment tooling)

**Goal.** Know what "bad" and "decent" look like before training anything.

**Hands-on:** 2 h environment + 2 h RL-side reading of results. **Training:** under 1 h of evaluation runs.

**Approach**

1. **Evaluation harness** (`evaluate.py`). Environment-side tooling, so code is fine. It takes any function that maps `(obs, mask)` to an action:

   ```python
   import numpy as np
   from xcom_sim import XcomEnv

   def evaluate(policy, n=500, first_seed=1_000_000, **env_kwargs):
       """Play n battles on held-out seeds. Never train on seeds >= first_seed."""
       env, wins, alive, turns = XcomEnv(**env_kwargs), [], [], []
       for k in range(n):
           obs, info = env.reset(seed=first_seed + k); done = False
           while not done:
               obs, r, term, trunc, info = env.step(policy(obs, info["action_mask"]))
               done = term or trunc
           wins.append(info["result"] == "win"); alive.append(info["soldiers_alive"]); turns.append(info["turn"])
       p = np.mean(wins); ci = 1.96 * np.sqrt(p * (1 - p) / n)
       return dict(win_rate=p, ci95=ci, soldiers_alive=np.mean(alive), turns=np.mean(turns))

   rng = np.random.default_rng(0)
   random_policy = lambda obs, mask: int(rng.choice(np.flatnonzero(mask)))
   print(evaluate(random_policy))
   ```

2. **Random baseline.** In my test runs of 500 battles it won roughly 10% (9.6% and 11.2% on two seed sets) and lost nearly all the rest, with a few timeouts. Reproduce this number on your machine.
3. **Scripted baseline** (`scripted.py`). A hand-written soldier: if it can shoot, take a snap shot at the nearest visible alien; otherwise step toward the far corner; end the turn when under about 15 TUs so it keeps some initiative. This needs access to the `Battle` object, not just the observation, which is fine for a baseline.
4. **Throughput.** Time random play and record steps per second. Then wrap several copies with `gymnasium.vector.AsyncVectorEnv` and time it again. On your CPU, 6 to 8 copies is a sensible range; more will compete with the training thread.

**Measures of success**

- A results table with win rate ± confidence interval, soldiers surviving, and turns taken, for both baselines.
- A measured steps-per-second figure, and updated training-time estimates.

---

## Phase 3: Your own masked PPO on 2 vs 2 (RL: your code)

**Goal.** Your hand-written PPO learns to beat the baselines on the smallest scenario.

**Hands-on:** about 10 h RL + 1 h environment fixes. **Training:** many short runs, 5–15 h total. A first sign of learning should appear within 30 to 60 minutes of a run; if not, stop and debug.

No code in this phase, by your rule. Files and what each must do:

**`agents/networks.py`**

- *Encoder.* Takes the `grid` tensor through two or three small convolution layers, flattens it, and joins it with the `unit` vector. On a 12 × 12 map this is tiny; keep total parameters in the tens of thousands at first.
- *Policy head.* Produces one score per action. A method takes the scores and the mask and returns a distribution in which illegal actions have zero probability.
- *Value head.* Produces a single number for the state. In this first version it sees the same observation as the policy.
- Your lab uses separate policy and value networks. A shared encoder with two heads is cheaper and usually works; either is acceptable. Pick one and note it.

**`agents/buffer.py`**

- Stores, for each step and each parallel environment: observation, mask, action, log-probability of the action, reward, done flag, and value estimate.
- A method computes returns and advantages once a rollout is complete. Start with your lab's discounted return. Then read about Generalized Advantage Estimation and add it as an option; it usually makes learning steadier.
- A method yields shuffled mini-batches, like the lab's `DataLoader` did.

**`agents/ppo.py`**

- *Collect.* Run the current policy for a fixed number of steps in every parallel environment, storing everything above. When an environment finishes it resets automatically; your returns must not leak across that boundary.
- *Update.* Your lab's `learn_ppo` with four additions: re-apply the stored mask when recomputing probabilities; normalise advantages within each mini-batch; add an entropy bonus so the policy does not collapse early; clip the gradient norm.
- *Log.* Episode reward, win rate, episode length, policy loss, value loss, entropy, and the fraction of probability ratios that were clipped.

**`train.py`**

- Reads a config file, sets every random seed, builds the parallel environments, runs collect and update in a loop, writes TensorBoard logs, saves checkpoints on a timer, and can resume from a checkpoint.
- Every so often it calls the Phase 2 evaluation on held-out seeds and logs the win rate.

**`env/rewards.py`** (your design)

- A function of `(events, result)` that returns a number. Start sparse. If nothing is learned after an hour, add small shaped terms: a little for damage dealt, a little less for damage taken, a little for the first sighting of an alien.
- Keep shaped terms much smaller than the win reward, or the agent will farm them. A classic failure: rewarding "alien spotted" every step teaches the agent to stare at an alien instead of shooting it.

**Starting hyperparameters** (common defaults; tune from here): learning rate 3e-4, discount 0.99, clip range 0.2, 8 environments × 128 steps per rollout, 4 update passes, mini-batches of 256, entropy weight 0.01.

**Order of work**

1. Make the environment trivially easy first: one soldier, one alien with 1 health standing in plain sight. Your PPO should reach nearly 100% wins in minutes. If it cannot, the bug is in your code, not in the difficulty.
2. Then 1 vs 1 with normal stats, then 2 vs 2.
3. Watch battles after each milestone using `env.render()`.

**"It won't learn" checklist**

1. Does it solve CartPole and the trivially easy scenario? If not, fix that first.
2. Is the mask applied both when acting and when updating? Is any illegal action ever chosen (`info["illegal"]`)?
3. Print tensor shapes and types once, as your lab advises. A silent broadcast between a `[N]` and a `[N, 1]` tensor is the most common PPO bug.
4. Is the advantage detached in the policy loss?
5. Is entropy falling to near zero in the first few updates? Raise the entropy weight.
6. Is the value loss huge compared with the policy loss? Scale rewards down or lower the value-loss weight.
7. Are returns leaking across episode boundaries in the buffer?
8. Watch three battles. Is the agent ending its turn immediately? That is often the safest thing it has found; add a small shaping term or a small per-turn cost.
9. Are you evaluating on the same seeds every time, and are they different from training seeds?

**Measures of success**

- Win rate clearly above the random baseline (the confidence intervals do not overlap).
- Win rate at or above the scripted baseline.
- Learning curves from at least 3 seeds showing the same trend.
- You can describe, from watching, one tactic the agent learned.

**Resources.** [GAE paper](https://arxiv.org/abs/1506.02438) · [37 PPO details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) · [CleanRL](https://docs.cleanrl.dev/) as a human-written reference; check your course's rules before reading reference implementations closely.

---

## Phase 4: Scale up and compare with a library

**Goal.** Show the approach survives bigger battles, and check your implementation against a trusted one.

**Hands-on:** about 8 h RL + 3 h environment. **Training:** 20–60 h total, mostly overnight.

**4a. Curriculum** (environment settings; the simulator already takes these as arguments)

| Stage | Soldiers vs aliens | Map | Move on when |
|---|---|---|---|
| 1 | 2 vs 2 | 12 × 12 | Phase 3 success |
| 2 | 3 vs 3 | 16 × 16 | Win rate above scripted baseline |
| 3 | 4 vs 4 | 20 × 20 | Same |
| 4 | 6 vs 8 | 30 × 30 | Stretch within the core plan |

Start each stage from the previous stage's weights. Two design problems appear here, both yours to solve:

- **The network must not depend on map size.** A flatten layer fixes the input size. Options: pad every map to the largest size from the start, or crop the grid to a fixed window centred on the active soldier.
- **The action list grows with the number of aliens.** Padding to a maximum alien count with those slots masked out is the simple answer.

**4b. Environment upgrades** (pick one or two; each needs a test and a baseline re-run)

- Kneeling: 4 TU to kneel, 8 to stand, +15% accuracy.
- Facing and a forward vision cone, with turning at 1 TU per 45°.
- Auto shots (rifle: 35% TU, 35% accuracy, three rounds).
- Randomised weapons.

**4c. Centralised critic.** Give the value head extra information the policy does not get, such as the true positions of all aliens. This is the multi-agent idea from our earlier discussion and needs one extra observation key from the environment (easy to add) plus your changes to the value network.

**4d. Library comparison.** Train [MaskablePPO from sb3-contrib](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) on the same environment. The provided `XcomEnv` already has the `action_masks()` method it looks for, and it supports dictionary observations and parallel environments. If the library does much better than your code on identical settings, you have a bug or a missing detail worth finding; that comparison is itself a good result to report. Confirm with your instructor whether library runs may count toward the project.

**Measures of success**

- A table of win rate ± CI per curriculum stage against both baselines.
- One ablation with a clear answer, for example sparse vs shaped reward, or with vs without the centralised critic.
- Your PPO and MaskablePPO within a few points of each other on 2 vs 2.

**Resources.** [MAPPO paper](https://arxiv.org/abs/2103.01955) · [Gym-µRTS paper](https://arxiv.org/abs/2105.13807) (PPO with masking on a grid strategy game) · [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/)

---

═══════════════════════════════════════════════════════════════

# STRETCH GOALS (beyond the 30-hour plan)

═══════════════════════════════════════════════════════════════

Everything below is optional. Phase 5 is cheap and worth doing if you have a spare evening. Phases 6 and 7 are each as large as the whole core plan.

## Phase 5: Build OpenXcom Extended and learn its battle code

**Goal.** A copy of OXCE you compiled yourself, and a map of where a bridge would attach.

**Hands-on:** 3–6 h. **Training:** none.

**Approach**

1. Install the released game first (OXCE 8.7.1, September 2026) from the [download thread](https://openxcom.org/forum/index.php?topic=5258.0), point it at your Steam data, and play a few rounds of **New Battle** from the main menu. New Battle lets you choose mission, terrain, alien race and equipment, which is exactly the scenario control you need.
   - Steam data is normally in `steamapps\common\XCom UFO Defense\XCOM` (folders such as `GEODATA`, `MAPS`, `ROUTES`, `TERRAIN`, `UNITS`). This path is from memory; confirm it on your machine.
2. Build from source, following the [official compiling thread](https://openxcom.org/forum/index.php?topic=7048.0). In short: install Visual Studio 2022 Community, clone [MeridianOXC/OpenXcom](https://github.com/MeridianOXC/OpenXcom/tree/oxce-plus) (branch `oxce-plus`), extract the pre-compiled dependencies linked in that thread into the repository root, open `src/OpenXcom.2010.sln`, and build. The thread describes this route as a 32-bit build, and it requires a C++17 compiler.
3. Read these, in this order (all confirmed in the current source):

   | File | What to look for |
   |---|---|
   | `src/Battlescape/BattlescapeGame.h` | The `BattleAction` struct: `actor`, `type`, `target`, `weapon`, and TU cost |
   | `src/Battlescape/BattlescapeGame.cpp` → `think()` | Runs when no animation is in progress. On a non-player side it calls `handleAI()` for the selected unit |
   | `handleAI()` | Asks the unit's `AIModule` for a `BattleAction`, then queues it: `BA_WALK` → pathfinding + `UnitWalkBState`; shots → `UnitTurnBState` + `ProjectileFlyBState` |
   | `primaryAction()` | How a human click becomes the same queued states |
   | `requestEndTurn()`, `tallyUnits()` | Ending a turn; counting living soldiers and aliens (your win/loss signal) |
   | `src/Battlescape/AIModule.cpp` | The alien decision logic |

4. Turn on `traceAI` in the options file to log what the alien AI is thinking.

**Useful to know**

- [Brutal-OXCE](https://github.com/Xilmi/OpenXcom/releases) is a fork with a much stronger AI and an **"Automated combat"** option that hands your own soldiers to the AI. That proves the engine can drive player units from code, gives you working source to study, and is a strong scripted baseline for the real game.
- I found **no headless mode** and **no existing Gym-style wrapper** for OpenXcom. Expect to turn animation speeds to maximum and, later, to skip drawing yourself.
- OpenXcom is GPL v3. A modified copy on your own machine is fine; if you publish it, your changes must be GPL too.

**Measure of success.** Your own build runs a New Battle, and you can point to the line where an AI action becomes a queued walk or shot.

---

## Phase 6: The bridge (environment code)

**Goal.** `OxceEnv`, a Gymnasium environment with the same observation and action format as the simulator, backed by the real game.

**Hands-on:** 15–30 h. This is the riskiest part of the whole project.

**Design.** The game is the server. When it is the player's turn and nothing is animating, it sends the battle state as one line of JSON over a local TCP socket and waits for one line back containing an action. The Python side turns that JSON into the same `grid` and `unit` arrays as the simulator.

The hook goes in the player-side branch of `BattlescapeGame::think()` and mirrors what `handleAI()` already does. This is an **untested sketch** to show the shape, not working code:

```cpp
// BattlescapeGame.cpp, inside think(), where _states.empty() and side == FACTION_PLAYER
if (Bridge::enabled() && _playerPanicHandled)
{
    BattleUnit *unit = _save->getSelectedUnit();
    if (!unit || unit->isOut()) { _save->selectNextPlayerUnit(); return ret; }

    BridgeCommand cmd = Bridge::exchange(serializeState(_save, unit));   // blocks until Python answers

    BattleAction action;
    action.actor = unit;
    if (cmd.kind == "walk")
    {
        action.type = BA_WALK;
        action.target = Position(cmd.x, cmd.y, cmd.z);
        _save->getPathfinding()->calculate(unit, action.target, BAM_NORMAL);
        if (_save->getPathfinding()->getStartDirection() != -1)
            statePushBack(new UnitWalkBState(this, action));
    }
    else if (cmd.kind == "snap" || cmd.kind == "aimed")
    {
        action.type = (cmd.kind == "snap") ? BA_SNAPSHOT : BA_AIMEDSHOT;
        action.weapon = unit->getMainHandWeapon();
        action.target = Position(cmd.x, cmd.y, cmd.z);
        action.updateTU();
        statePushBack(new UnitTurnBState(this, action));
        statePushBack(new ProjectileFlyBState(this, action));
    }
    else if (cmd.kind == "next_unit") { _save->selectNextPlayerUnit(); }
    else if (cmd.kind == "end_turn")  { requestEndTurn(false); }
}
```

`serializeState` walks `_save->getUnits()` and the tile grid and writes positions, TUs, health, faction and visibility. `Bridge` is a small class you add (about 100 lines of Winsock).

**Work order**

1. One-way first: dump the state to a file each turn and draw it with your Python renderer. No control yet.
2. Add the socket and drive one soldier with random legal moves.
3. Add shots, end of turn, end of battle detection, and automatic restart into a new battle.
4. Speed: set all battle animation speeds to maximum in the options, then look at skipping drawing.

**Measures of success.** The random policy completes 50 battles in a row unattended; you have a steps-per-second figure; and the scripted baseline's win rate in the real game is measured.

**Biggest risks.** Pop-up dialogs that wait for a click (panic messages, end-of-mission confirmation); mismatches between what the simulator and the engine allow; and speed, since the engine may be 10 to 100 times slower than the simulator.

---

## Phase 7: The full game

In rough order of difficulty, once the bridge works. Each one is a project in itself.

1. **Transfer.** Fine-tune the simulator-trained network in the real game. Expect a drop; the closer the two observation formats and rule sets, the smaller it is.
2. **Real maps and multiple floors**, including landed UFOs. The observation gains a height dimension.
3. **Larger squads** up to 14, and letting the policy choose which soldier acts next.
4. **More aliens and weapons.** [Chryssalids](https://www.ufopaedia.org/Chryssalid) turn soldiers into zombies; [Ethereals](https://www.ufopaedia.org/Ethereal) use psionics, which the agent cannot see coming. Stronger opponents via Brutal-OXCE.
5. **Night missions** (shorter sight range, flares), **civilians** in [terror missions](https://www.ufopaedia.org/Terror_Mission), [morale](https://www.ufopaedia.org/Morale) and panic.
6. **Inventory**: grenades, reloading, spare weapons. This multiplies the action space and probably needs a different action design.

Note that the real alien AI gets to know every soldier's position after turn 20 ([Tactical AI](https://www.ufopaedia.org/Tactical_AI), [reaction fire triggers](https://www.ufopaedia.org/Reaction_fire_triggers)), so slow play is punished.

---

## 5. Measures of success for the whole project

Report every number as an average over 500 held-out battles with a 95% confidence interval.

| Measure | Notes |
|---|---|
| **Win rate** (primary) | Always next to the random and scripted baselines for the same scenario |
| Soldiers surviving | And their average remaining health |
| Turns to finish | Lower is better, but only among wins |
| Timeouts | A high rate means the agent is hiding, not fighting |
| Civilians surviving | Stretch scenarios only |
| Learning curves | Win rate against environment steps, at least 3 seeds |
| Progress evidence | Git history, run logs, and a short note per experiment: what changed, what happened |

Since the project is judged on progress, the record of what you tried and what you learned from failures is part of the result. Keep a one-line-per-run log from day one.

## 6. Risks

| Risk | Likelihood | What to do |
|---|---|---|
| PPO works on CartPole but not on X-COM | High at first | The trivially easy scenario in Phase 3, then the checklist |
| Agent learns to hide and time out | High | Small per-turn cost or shaping; watch the timeout rate |
| Reward shaping gets exploited | Medium | Keep shaped terms small; always judge by win rate, never by reward |
| Training too slow on the laptop | Medium | Smaller maps, fewer pass-throughs, parallel environments; profile before guessing |
| A simulator bug invalidates results | Medium | Tests for every rule; re-run baselines after every environment change |
| OXCE build or bridge eats the schedule | High | It is below the line on purpose |

## 7. Glossary

**RL terms.** *Policy:* the network that picks actions. *Value:* a prediction of future reward from a state. *Advantage:* how much better an action turned out than the value predicted. *Rollout:* a batch of experience collected with the current policy. *Action mask:* the list of currently legal actions. *Reward shaping:* extra small rewards that guide learning toward the real goal. *Curriculum:* training on easy versions first. *Centralised critic:* a value network that sees more than the policy does. *Baseline:* a simple non-learning policy to compare against.

**X-COM terms.** *Battlescape:* the tactical battle mode. *Time Units (TU):* a unit's action budget per turn. *Snap / aimed / auto shot:* fast and inaccurate, slow and accurate, or a three-round burst. *Reaction fire:* shooting during the enemy's turn. *Sectoid:* the weakest common alien. *Rookie:* a new soldier with random low stats. *OXCE:* OpenXcom Extended.

## 8. Resources

**Game and engine**
- [OpenXcom](https://openxcom.org/) · [OXCE source, branch oxce-plus](https://github.com/MeridianOXC/OpenXcom/tree/oxce-plus) · [OXCE downloads](https://openxcom.org/forum/index.php?topic=5258.0) · [Compiling OXCE](https://openxcom.org/forum/index.php?topic=7048.0)
- [Brutal-OXCE releases](https://github.com/Xilmi/OpenXcom/releases) · [Brutal-OXCE forum thread](https://openxcom.org/forum/index.php?topic=11663.0)
- [OpenXcom class documentation: BattlescapeGame](https://openxcom.org/docs/html/class_open_xcom_1_1_battlescape_game.html) (older, but the structure holds)

**Rules (UFOpaedia)**
- [Time Units](https://www.ufopaedia.org/Time_Units) · [Walking costs](https://www.ufopaedia.org/Walking_Costs_(UFO)) · [Firing Accuracy](https://www.ufopaedia.org/Firing_Accuracy) · [Accuracy formula](https://www.ufopaedia.org/Accuracy_formula) · [Reaction fire](https://www.ufopaedia.org/Reaction_fire) · [Reaction fire triggers](https://www.ufopaedia.org/Reaction_fire_triggers) · [Line of sight](https://www.ufopaedia.org/Line_of_sight) · [Damage](https://www.ufopaedia.org/Damage)
- [Sectoid](https://www.ufopaedia.org/Sectoid) · [Alien Stats](https://www.ufopaedia.org/Alien_Stats) · [Rifle](https://www.ufopaedia.org/Rifle) · [Plasma Pistol](https://www.ufopaedia.org/Plasma_Pistol) · [Tactical AI](https://www.ufopaedia.org/Tactical_AI)

**Reinforcement learning**
- [Spinning Up in Deep RL](https://spinningup.openai.com/) · [Sutton and Barto, free textbook](http://incompleteideas.net/book/the-book-2nd.html) · [Hugging Face Deep RL course](https://huggingface.co/learn/deep-rl-course)
- [PPO paper](https://arxiv.org/abs/1707.06347) · [37 PPO implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) · [GAE](https://arxiv.org/abs/1506.02438) · [Invalid action masking](https://arxiv.org/abs/2006.14171) · [MAPPO](https://arxiv.org/abs/2103.01955) · [Gym-µRTS](https://arxiv.org/abs/2105.13807)
- [Gymnasium](https://gymnasium.farama.org/) · [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) · [sb3-contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) · [CleanRL](https://docs.cleanrl.dev/)

## 9. What was and was not verified

- **Checked against the source this week:** the OXCE version and build steps, the `BattlescapeGame` functions and `BattleAction` fields named in Phase 5, Brutal-OXCE's automated combat option, MaskablePPO's masking interface, and every UFOpaedia number in section 4 not marked "from memory".
- **From memory, check before relying on it:** rookie health and reactions ranges, the 20-tile sight range, the 0–200% damage range, the Steam data folder path, and the rifle's auto-shot round count.
- **Not tested:** the C++ bridge sketch, and all training-time estimates. The simulator and its tests were run; no learning code was.
