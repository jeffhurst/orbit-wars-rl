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
