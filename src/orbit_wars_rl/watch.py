"""Watch a trained Orbit Wars RL model play against a random bot."""

from __future__ import annotations

import argparse
import os
import webbrowser
from pathlib import Path

from kaggle_environments import make

from orbit_wars_rl.opponents.random_bot import agent as random_agent
from orbit_wars_rl.submit_agent import agent as rl_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained MaskablePPO .zip model.",
    )
    parser.add_argument(
        "--out",
        default="replays/orbit_wars_replay.html",
        help="HTML replay output path.",
    )
    parser.add_argument(
        "--rl-player",
        type=int,
        choices=[0, 1],
        default=0,
        help="Which player the RL agent controls.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Kaggle env debug mode.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # submit_agent.py reads this when it loads the model.
    os.environ["ORBIT_WARS_MODEL"] = str(model_path)

    env = make("orbit_wars", debug=args.debug)

    if args.rl_player == 0:
        agents = [rl_agent, random_agent]
    else:
        agents = [random_agent, rl_agent]

    env.run(agents)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = env.render(mode="html")
    out_path.write_text(html, encoding="utf-8")

    print(f"Saved replay to: {out_path.resolve()}")

    for i, state in enumerate(env.state):
        print(
            f"Player {i}: "
            f"status={getattr(state, 'status', None)}, "
            f"reward={getattr(state, 'reward', None)}"
        )

    webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
