"""Fetch building footprints from Datafordeler GEODKV (GeoDanmark) API.

Ported from web-main: amsterdam/libraries/datafordeleren/src/bygning.ts
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

import requests

logger = logging.getLogger(__name__)

GEODKV_ENDPOINT = "https://graphql.datafordeler.dk/GEODKV/v1"
BBOX_SIZE_METERS = 20

# Statuses to exclude (not yet built)
_EXCLUDED_STATUSES = {"Projekt godkendt", "Under anlæg"}

BYGNING_QUERY = """
query GetBygninger(
    $where: GEODKV_BygningFilterInput
    $registreringstid: DafDateTime
    $virkningstid: DafDateTime
) {
    GEODKV_Bygning(
        first: 100
        where: $where
        registreringstid: $registreringstid
        virkningstid: $virkningstid
    ) {
        nodes {
            geometri { wkt }
            bygningstype
            status
            id_lokalId
            BBRUUID
        }
    }
}
"""


def _wgs84_to_utm(lng: float, lat: float) -> tuple[float, float]:
    """Convert WGS84 (lng, lat) to EPSG:25832 (easting, northing).

    Uses pyproj if available, otherwise falls back to a simplified formula.
    """
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        return transformer.transform(lng, lat)
    except ImportError:
        import math

        # Simplified UTM zone 32N conversion (good enough for Denmark)
        k0 = 0.9996
        a = 6378137.0
        e = 0.0818191908426
        lon0 = math.radians(9.0)  # central meridian for zone 32
        lat_r = math.radians(lat)
        lon_r = math.radians(lng)
        n = a / math.sqrt(1 - e**2 * math.sin(lat_r) ** 2)
        t = math.tan(lat_r) ** 2
        c = (e**2 / (1 - e**2)) * math.cos(lat_r) ** 2
        aa = (lon_r - lon0) * math.cos(lat_r)
        m = a * (
            (1 - e**2 / 4 - 3 * e**4 / 64) * lat_r
            - (3 * e**2 / 8 + 3 * e**4 / 32) * math.sin(2 * lat_r)
            + (15 * e**4 / 256) * math.sin(4 * lat_r)
        )
        easting = k0 * n * (aa + (1 - t + c) * aa**3 / 6) + 500000
        northing = k0 * (
            m + n * math.tan(lat_r) * (aa**2 / 2 + (5 - t + 9 * c) * aa**4 / 24)
        )
        return easting, northing


def _utm_to_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """Convert EPSG:25832 (easting, northing) to WGS84 (lng, lat)."""
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        return transformer.transform(easting, northing)
    except ImportError:
        import math

        # Simplified inverse UTM for zone 32N (adequate for Denmark)
        k0 = 0.9996
        a = 6378137.0
        e = 0.0818191908426
        e1sq = e**2 / (1 - e**2)
        lon0 = math.radians(9.0)
        m = northing / k0
        mu = m / (a * (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256))
        e1 = (1 - math.sqrt(1 - e**2)) / (1 + math.sqrt(1 - e**2))
        phi1 = mu + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        phi1 += (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        phi1 += (151 * e1**3 / 96) * math.sin(6 * mu)
        n1 = a / math.sqrt(1 - e**2 * math.sin(phi1) ** 2)
        t1 = math.tan(phi1) ** 2
        c1 = e1sq * math.cos(phi1) ** 2
        r1 = a * (1 - e**2) / (1 - e**2 * math.sin(phi1) ** 2) ** 1.5
        d = (easting - 500000) / (n1 * k0)
        lat = phi1 - (n1 * math.tan(phi1) / r1) * (
            d**2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e1sq) * d**4 / 24
        )
        lng_rad = lon0 + (d - (1 + 2 * t1 + c1) * d**3 / 6) / math.cos(phi1)
        return math.degrees(lng_rad), math.degrees(lat)


def _build_bbox_wkt(lat: float, lng: float) -> str:
    """Build a 20m bounding box WKT polygon in EPSG:25832 around WGS84 coords."""
    center_e, center_n = _wgs84_to_utm(lng, lat)
    half = BBOX_SIZE_METERS / 2
    min_e = center_e - half
    min_n = center_n - half
    max_e = center_e + half
    max_n = center_n + half
    return (
        f"POLYGON(({min_e} {min_n}, {max_e} {min_n}, "
        f"{max_e} {max_n}, {min_e} {max_n}, "
        f"{min_e} {min_n}))"
    )


def _parse_wkt_polygon(wkt: str) -> list[list[tuple[float, float]]] | None:
    """Parse WKT POLYGON or MULTIPOLYGON into rings of (lng, lat) coords."""
    polygon_match = re.match(r"POLYGON\s*Z?\s*\(\((.+?)\)\)", wkt, re.IGNORECASE)
    multi_match = re.match(r"MULTIPOLYGON\s*Z?\s*\(\(\((.+?)\)\)\)", wkt, re.IGNORECASE)

    if polygon_match:
        return [_parse_ring(polygon_match.group(1))]

    if multi_match:
        parts = multi_match.group(1).split(")),((")
        return [_parse_ring(part) for part in parts]

    return None


def _parse_ring(coord_str: str) -> list[tuple[float, float]]:
    """Parse a coordinate string into a list of (lng, lat) tuples."""
    coords = []
    for pair in coord_str.split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                easting, northing = float(parts[0]), float(parts[1])
                lng, lat = _utm_to_wgs84(easting, northing)
                coords.append((lng, lat))
            except (ValueError, Exception):
                continue
    return coords


def fetch_building_footprints(lat: float, lng: float, api_key: str) -> dict:
    """Fetch building footprints from Datafordeler GEODKV API.

    Returns a GeoJSON FeatureCollection with Polygon features in WGS84.
    Properties per feature: bygningstype, status, id_lokalId, BBRUUID.

    Matches web-main getBuildingFootprints() in bygning.ts.
    """
    filter_wkt = _build_bbox_wkt(lat, lng)
    now = datetime.now(UTC).isoformat()

    url = f"{GEODKV_ENDPOINT}?apiKey={requests.utils.quote(api_key)}"
    response = requests.post(
        url,
        json={
            "query": BYGNING_QUERY,
            "variables": {
                "where": {
                    "geometri": {
                        "intersects": {"crs": 25832, "wkt": filter_wkt},
                    },
                },
                "registreringstid": now,
                "virkningstid": now,
            },
        },
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    response.raise_for_status()

    body = response.json()
    if body.get("errors"):
        msgs = ", ".join(e.get("message", "") for e in body["errors"])
        raise RuntimeError(f"Datafordeler GraphQL errors: {msgs}")

    data = body.get("data")
    if not data:
        raise RuntimeError("Datafordeler GraphQL response has no data")

    nodes = (data.get("GEODKV_Bygning") or {}).get("nodes") or []

    features = []
    for node in nodes:
        wkt = (node.get("geometri") or {}).get("wkt")
        status = node.get("status")
        bygningstype = node.get("bygningstype")

        if not wkt or not status or not bygningstype:
            continue
        if status in _EXCLUDED_STATUSES:
            continue

        rings = _parse_wkt_polygon(wkt)
        if not rings:
            continue

        properties = {
            "bygningstype": bygningstype,
            "status": status,
            "id_lokalId": node.get("id_lokalId"),
            "BBRUUID": node.get("BBRUUID"),
        }

        # Each ring becomes a separate Feature (matches web-main behavior)
        for ring in rings:
            if len(ring) < 3:
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[lng, lat] for lng, lat in ring]],
                    },
                }
            )

    return {"type": "FeatureCollection", "features": features}


def fetch_building_footprints_as_osm_format(
    lat: float, lng: float, api_key: str, gps: dict | None = None
) -> list[dict]:
    """Fetch from Datafordeler and return in the same format as OSM features.

    This allows drop-in replacement of osm_on_address_features in the viewer.
    Each feature has: type, geometry, properties (with id_lokalId, BBRUUID,
    centroid_wgs84, data_source).
    """
    fc = fetch_building_footprints(lat, lng, api_key)

    features = []
    for f in fc.get("features", []):
        geom = f.get("geometry", {})
        props = dict(f.get("properties", {}))
        props["data_source"] = "datafordeler"

        # Compute centroid from polygon coordinates
        coords = geom.get("coordinates", [[]])[0]
        if coords:
            clng = sum(c[0] for c in coords) / len(coords)
            clat = sum(c[1] for c in coords) / len(coords)
            props["centroid_wgs84"] = {"lat": clat, "lng": clng}

        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": props,
            }
        )

    return features
