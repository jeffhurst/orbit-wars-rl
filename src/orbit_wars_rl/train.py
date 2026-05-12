"""Train MaskablePPO on OrbitWarsGym."""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO

from orbit_wars_rl.envs.orbit_wars_gym import OrbitWarsGym
from orbit_wars_rl.opponents.greedy_bot import agent as greedy_agent
from orbit_wars_rl.opponents.random_bot import agent as random_agent
from orbit_wars_rl.opponents.starter_bot import agent as starter_agent
from orbit_wars_rl.utils.paths import MODELS_DIR, RUNS_DIR, ensure_dir
from orbit_wars_rl.utils.seeds import set_global_seeds
from orbit_wars_rl.utils.tensorboard_callback import CustomTensorboardCallback

OPPONENTS = {"random": random_agent, "starter": starter_agent, "greedy": greedy_agent}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Orbit Wars MaskablePPO candidate policy.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--opponent", choices=OPPONENTS.keys(), default="starter")
    parser.add_argument("--save-path", type=Path, default=MODELS_DIR / "orbit_wars_maskableppo_v1.zip")
    parser.add_argument("--tensorboard-log", type=Path, default=RUNS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    ensure_dir(args.save_path.parent)
    ensure_dir(args.tensorboard_log)

    env = OrbitWarsGym(opponent_agent=OPPONENTS[args.opponent], player_id=0)
    env.reset(seed=args.seed)
    env.require_real_kaggle_env()

    print(f"Training MaskablePPO for {args.timesteps:,} timesteps against {args.opponent} opponent")
    tensorboard_callback = CustomTensorboardCallback()
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=128,
        batch_size=128,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=2.5e-4,
        ent_coef=0.01,
        tensorboard_log=str(args.tensorboard_log),
        seed=args.seed,
        verbose=1,
    )
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=False,
        callback=tensorboard_callback,
    )
    model.save(str(args.save_path))
    print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()
