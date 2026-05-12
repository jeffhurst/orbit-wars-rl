from types import SimpleNamespace

from orbit_wars_rl.features.metrics import count_captures_and_losses, summarize_planets


def test_metrics_accept_dict_and_attribute_planet_records():
    previous_obs = SimpleNamespace(
        player=0,
        planets=[
            {
                "id": 0,
                "owner": 0,
                "x": 0,
                "y": 0,
                "radius": 1,
                "ships": 10,
                "production": 2,
            },
            SimpleNamespace(id=1, owner=-1, x=1, y=1, radius=1, ships=5, production=1),
        ],
    )
    current_obs = SimpleNamespace(
        player=0,
        planets=[
            {
                "id": 0,
                "owner": 1,
                "x": 0,
                "y": 0,
                "radius": 1,
                "ships": 8,
                "production": 2,
            },
            SimpleNamespace(id=1, owner=0, x=1, y=1, radius=1, ships=7, production=1),
        ],
    )

    summary = summarize_planets(current_obs)
    ownership_changes = count_captures_and_losses(previous_obs, current_obs)

    assert summary["planet_count_mine"] == 1.0
    assert summary["planet_count_enemy"] == 1.0
    assert ownership_changes == {"captures": 1.0, "planets_lost": 1.0}
