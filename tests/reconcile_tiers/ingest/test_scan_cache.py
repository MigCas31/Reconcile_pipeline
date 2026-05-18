from pathlib import Path

from reconcile_tiers.ingest.scan_cache import (
    find_scan_cache_dir,
    load_raw_ceilings,
    load_raw_rooms,
    parse_address_from_scan_dir,
)


def test_find_scan_cache_dir_and_parse_address():
    uuid = "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd"

    scan_dir = find_scan_cache_dir(uuid, Path(".scan-cache"))

    assert scan_dir is not None
    assert scan_dir.name.endswith(f"{uuid}_1773314869.0916429")
    assert parse_address_from_scan_dir(scan_dir) == "Kildegaardsvej 8 5856 Ryslinge"


def test_load_raw_rooms_excludes_metadata_and_ceilings():
    uuid = "c72ad855-9e52-46f1-886d-a9f37911521f"
    scan_dir = find_scan_cache_dir(uuid, Path(".scan-cache"))

    rooms = load_raw_rooms(scan_dir)

    assert len(rooms) == 10
    filenames = [filename for filename, _room in rooms]
    assert filenames == sorted(filenames)
    assert "data.json" not in filenames
    assert "arworldmap.json" not in filenames
    assert not any(filename.startswith("ceiling_") for filename in filenames)


def test_load_raw_ceilings_reads_all_planes_and_metadata_source():
    uuid = "c72ad855-9e52-46f1-886d-a9f37911521f"
    scan_dir = find_scan_cache_dir(uuid, Path(".scan-cache"))

    ceilings = load_raw_ceilings(scan_dir)

    assert len(ceilings) == 10
    assert set(ceilings).issubset(
        {filename for filename, _room in load_raw_rooms(scan_dir)}
    )
    assert all(payload["planes"] for payload in ceilings.values())
    first = next(iter(ceilings.values()))
    assert set(first) == {"planes", "source"}
    assert {"corners_local", "transform"} <= set(first["planes"][0])
