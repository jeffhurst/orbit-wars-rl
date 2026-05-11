from orbit_wars_rl.features.action_decoder import decode_candidate


def test_action_decoder_returns_empty_for_invalid_index():
    assert decode_candidate(99, [{"type": "noop"}]) == []


def test_action_decoder_returns_empty_for_noop():
    assert decode_candidate(0, [{"type": "noop"}]) == []


def test_action_decoder_decodes_send():
    assert decode_candidate(0, [{"type": "send", "from_planet_id": 1, "angle": 0.5, "ships": 3}]) == [[1, 0.5, 3]]
