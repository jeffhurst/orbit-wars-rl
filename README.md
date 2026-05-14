# orbit-wars-rl

A clean reinforcement-learning training project for the Kaggle **Orbit Wars** competition.

The project is intentionally built around a discrete candidate-action interface:

1. helper code generates legal/plausible Orbit Wars actions,
2. `MaskablePPO` chooses one candidate index,
3. the selected candidate is decoded into the Kaggle action format.

The goal is a serious RL pipeline, not a hidden rule-based strategic bot. The v1 policy does **not** train raw continuous actions.

## Install

Use Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest
```

The environment wrapper includes a tiny synthetic fallback for smoke tests when `kaggle-environments` is unavailable, but real training requires Kaggle's `orbit_wars` environment.

## Train

```bash
python -m orbit_wars_rl.train --timesteps 10000 --opponent starter
```

Useful arguments:

```bash
python -m orbit_wars_rl.train \
  --timesteps 100000 \
  --opponent starter \
  --save-path models/orbit_wars_maskableppo_v1.zip \
  --tensorboard-log runs \
  --seed 42
```

Available opponents: `random`, `starter`, `greedy`.

PowerShell helper:

```powershell
.\scripts\train_v1.ps1 -Timesteps 100000 -Opponent starter
```

## Evaluate

```bash
python -m orbit_wars_rl.evaluate --model-path models/orbit_wars_maskableppo_v1.zip --games 10 --opponent starter
```

PowerShell helper:

```powershell
.\scripts\evaluate_v1.ps1 -ModelPath models/orbit_wars_maskableppo_v1.zip -Games 10 -Opponent starter
```

Evaluation reports wins, losses, draws when raw Kaggle rewards are available, plus average shaped reward and average final score when detectable.

For repeated evaluations across distinct seeds, choose a base `--seed`; `evaluate.py` resets game `n` with `seed + n`, so `--games 10 --seed 1000` evaluates seeds `1000` through `1009`:

```bash
python -m orbit_wars_rl.evaluate \
  --model-path models/orbit_wars_maskableppo_v1.zip \
  --games 10 \
  --opponent starter \
  --seed 1000
```

To compare multiple Orbit Wars map presets or map-generation options, run separate evaluation batches with the same seed range and pass the map data through Gymnasium reset options when constructing a custom evaluation loop. `OrbitWarsGym.reset(seed=..., options=...)` forwards the seed through Kaggle construction/reset paths that accept it, preserves unsupported APIs, and returns `info["seed"]` plus `info["kaggle_seed_applied"]` so logs can confirm whether the real Kaggle environment received the seed:

```python
from orbit_wars_rl.envs.orbit_wars_gym import OrbitWarsGym
from orbit_wars_rl.opponents.starter_bot import agent as starter_agent

for map_name in ["map-a", "map-b"]:
    for seed in range(1000, 1010):
        env = OrbitWarsGym(opponent_agent=starter_agent)
        obs, info = env.reset(seed=seed, options={"map": map_name})
        print(map_name, seed, info["kaggle_seed_applied"])
        env.close()
```


## Diagnosing flat training runs

If TensorBoard shows `rollout/ep_rew_mean` stuck far below zero after a few hundred thousand steps, do not treat that scalar as a win-rate measurement. It is the sum of shaped per-turn rewards over long 500-turn games, so it can be strongly negative even when individual tactical choices improve. Always run `evaluate.py` and compare wins/losses/draws against the target opponent.

For runs that fail to beat the random opponent, the first things to inspect are:

- `custom/noop_rate`, `custom/send_rate`, and `custom/invalid_action_rate`: high NOOP or invalid rates mean the policy is not using the candidate mask/action list effectively.
- `custom/avg_candidates`: very small candidate counts usually mean the observation parser does not recognize owned planets or the agent has already lost map control.
- `custom/target_owner_neutral_rate`, `custom/target_owner_enemy_rate`, and `custom/target_owner_self_rate`: a policy that mostly reinforces itself or never attacks/captures will not beat random play.
- `train/approx_kl`: values that stay near zero while entropy remains high usually mean PPO is making tiny updates and needs more signal, a larger training budget, or easier curriculum/opponent settings.

The candidate observation rows include explicit send/target-type/source-position/target-position features. Older checkpoints trained with the previous 6-feature candidate rows are not compatible with newly trained policies using the current observation shape.

## TensorBoard

```bash
tensorboard --logdir runs
```

## Project architecture

```text
src/orbit_wars_rl/
  envs/orbit_wars_gym.py          Gymnasium wrapper around kaggle_environments.make("orbit_wars")
  features/observation_encoder.py Fixed-size defensive observation encoder
  features/candidate_generator.py Candidate action generation; candidate 0 is always NOOP
  features/action_decoder.py      Safe candidate-index to Kaggle-action decoder
  features/reward_shaping.py      Small terminal + advantage-delta reward shaping
  opponents/                     Simple random/starter/greedy baselines
  train.py                       MaskablePPO training entry point
  evaluate.py                    Saved-model evaluation entry point
  submit_agent.py                Kaggle-compatible agent with model-load fallback
```

## Orbit Wars assumptions inspected

The current Kaggle environment documentation describes:

- planets as `[id, owner, x, y, radius, ships, production]`,
- fleets as `[id, owner, x, y, angle, from_planet_id, ships]`,
- player id in `player`,
- actions as `[[from_planet_id, direction_angle, num_ships], ...]`,
- a 100x100 board and 500-turn default games.

The code still parses defensively so it can tolerate struct-like observations, missing fields, extra fields, and future minor format differences.

## Current limitations

- v1 supports one learning player in a 2-player match.
- Candidate actions aim at current target positions; no intercept prediction yet.
- Reward shaping is deliberately small and simple.
- Opponent bots are simple baselines used for curriculum smoke training, not strong strategic agents.
- The submission agent expects a compatible model file at `models/orbit_wars_maskableppo_v1.zip` or a path in `ORBIT_WARS_MODEL`.

## Next steps

- Add vectorized environments once mask handling is verified end to end.
- Add 4-player support in `OrbitWarsGym`.
- Add orbital/intercept-aware candidate generation.
- Expand observation features for comets and predicted planet positions.
- Add opponent sampling/curriculum during training.
- Package `submit_agent.py` and model artifacts into a Kaggle submission bundle.
