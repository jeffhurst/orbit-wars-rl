"""Train MaskablePPO on OrbitWarsGym."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor

from orbit_wars_rl.envs.orbit_wars_gym import OrbitWarsGym
from orbit_wars_rl.opponents.greedy_bot import agent as greedy_agent
from orbit_wars_rl.opponents.random_bot import agent as random_agent
from orbit_wars_rl.opponents.starter_bot import agent as starter_agent
from orbit_wars_rl.utils.paths import MODELS_DIR, RUNS_DIR, ensure_dir
from orbit_wars_rl.utils.seeds import set_global_seeds
from orbit_wars_rl.utils.tensorboard_callback import CustomTensorboardCallback

OPPONENTS = {"random": random_agent, "starter": starter_agent, "greedy": greedy_agent}
DEFAULT_N_ENVS = 4
DEFAULT_N_STEPS = 512
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 2.5e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Orbit Wars MaskablePPO candidate policy."
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--opponent", choices=OPPONENTS.keys(), default="starter")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=MODELS_DIR / "orbit_wars_maskableppo_v1.zip",
    )
    parser.add_argument("--tensorboard-log", type=Path, default=RUNS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=DEFAULT_N_ENVS,
        help="Number of parallel OrbitWarsGym environments to collect rollouts from.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help="Steps per environment in each PPO rollout.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Minibatch size used by PPO updates.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="PPO optimizer learning rate.",
    )
    return parser.parse_args()


def make_orbit_wars_env(
    *, opponent: str, player_id: int, seed: int, env_index: int
) -> Callable[[], gym.Env]:
    """Create an OrbitWarsGym factory with a deterministic, distinct seed."""
    env_seed = seed + env_index

    def _init() -> OrbitWarsGym:
        env = OrbitWarsGym(opponent_agent=OPPONENTS[opponent], player_id=player_id)
        env.reset(seed=env_seed)
        env.require_real_kaggle_env()
        return env

    return _init


def build_vec_env(args: argparse.Namespace) -> VecEnv:
    """Build an SB3 vectorized env that exposes OrbitWarsGym.action_masks()."""
    env_fns = [
        make_orbit_wars_env(
            opponent=args.opponent,
            player_id=0,
            seed=args.seed,
            env_index=env_index,
        )
        for env_index in range(args.n_envs)
    ]
    return VecMonitor(DummyVecEnv(env_fns))


def validate_args(args: argparse.Namespace) -> None:
    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")
    if args.n_steps < 1:
        raise ValueError("--n-steps must be at least 1")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0")

    rollout_size = args.n_steps * args.n_envs
    if rollout_size <= 1:
        raise ValueError("--n-steps * --n-envs must be greater than 1 for PPO")
    if args.batch_size > rollout_size:
        raise ValueError(
            "--batch-size must be less than or equal to the effective rollout size "
            f"({rollout_size})"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_global_seeds(args.seed)
    ensure_dir(args.save_path.parent)
    ensure_dir(args.tensorboard_log)

    rollout_size = args.n_steps * args.n_envs
    print(
        f"Training MaskablePPO for {args.timesteps:,} timesteps "
        f"against {args.opponent} opponent"
    )
    print(
        "Rollout configuration: "
        f"n_envs={args.n_envs}, n_steps={args.n_steps}, "
        f"effective_rollout_size={rollout_size}, batch_size={args.batch_size}"
    )

    env = build_vec_env(args)
    try:
        tensorboard_callback = CustomTensorboardCallback()
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.995,
            gae_lambda=0.95,
            learning_rate=args.learning_rate,
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
    finally:
        env.close()


if __name__ == "__main__":
    main()
