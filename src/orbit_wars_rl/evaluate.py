"""Evaluate a saved MaskablePPO Orbit Wars model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from orbit_wars_rl.envs.orbit_wars_gym import OrbitWarsGym
from orbit_wars_rl.opponents.greedy_bot import agent as greedy_agent
from orbit_wars_rl.opponents.random_bot import agent as random_agent
from orbit_wars_rl.opponents.starter_bot import agent as starter_agent
from orbit_wars_rl.utils.seeds import set_global_seeds

OPPONENTS = {"random": random_agent, "starter": starter_agent, "greedy": greedy_agent}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Orbit Wars MaskablePPO model.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--opponent", choices=OPPONENTS.keys(), default="starter")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    model = MaskablePPO.load(str(args.model_path))
    rewards: list[float] = []
    scores: list[float] = []
    wins = losses = draws = 0

    for game in range(args.games):
        env = OrbitWarsGym(opponent_agent=OPPONENTS[args.opponent], player_id=0)
        obs, _ = env.reset(seed=args.seed + game)
        done = False
        total_reward = 0.0
        final_score = None
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            done = terminated or truncated
            final_score = info.get("final_score", final_score)
            steps += 1
            if steps > 600:
                break
        rewards.append(total_reward)
        if final_score is not None:
            scores.append(float(final_score))
        raw_reward = getattr(env._env.state[0], "reward", None) if env._env is not None else final_score
        opponent_reward = getattr(env._env.state[1], "reward", None) if env._env is not None else None
        if raw_reward is not None and opponent_reward is not None:
            if raw_reward > opponent_reward:
                wins += 1
            elif raw_reward < opponent_reward:
                losses += 1
            else:
                draws += 1
        env.close()

    print(f"Games: {args.games}")
    print(f"Opponent: {args.opponent}")
    print(f"Wins/Losses/Draws: {wins}/{losses}/{draws}")
    print(f"Average reward: {float(np.mean(rewards)) if rewards else 0.0:.3f}")
    print(f"Average final score: {float(np.mean(scores)) if scores else float('nan'):.3f}")


if __name__ == "__main__":
    main()
