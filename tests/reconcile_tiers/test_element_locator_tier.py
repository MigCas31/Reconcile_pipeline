from reconcile.element_locator import find_element


def test_tier_room_locator_accepts_floor_piece_list():
    uuid = "00000000-0000-0000-0000-000000000123"
    locator = f"{uuid}::tier-room::2"
    payload = {
        "uuid": uuid,
        "rooms": [
            {
                "locator_id": locator,
                "story": 0,
                "floor": [
                    {
                        "corners": [
                            {"x": 0.0, "y": 0.0, "z": 0.0},
                            {"x": 1.0, "y": 0.0, "z": 0.0},
                            {"x": 1.0, "y": 0.0, "z": 1.0},
                            {"x": 0.0, "y": 0.0, "z": 1.0},
                        ]
                    }
                ],
                "walls": [],
            }
        ],
    }

    result = find_element([], locator, tier_payloads={uuid: payload})

    assert result["json_path"] == "rooms[0].floor[0]"
    assert result["corners_count"] == 4
    assert result["element"]["id"] == locator
