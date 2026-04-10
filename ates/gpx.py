from __future__ import annotations

import io
from typing import BinaryIO

import gpxpy


def parse_gpx(file_obj: BinaryIO | bytes) -> list[dict]:
    """
    Parse a GPX file and return a flat list of track points.

    Returns a list of dicts with keys: lat, lon, ele (elevation in metres,
    or None if not present in the GPX).
    """
    if isinstance(file_obj, bytes):
        file_obj = io.BytesIO(file_obj)

    gpx = gpxpy.parse(file_obj)

    points: list[dict] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append(
                    {
                        "lat": pt.latitude,
                        "lon": pt.longitude,
                        "ele": pt.elevation,
                    }
                )

    # Fall back to routes if no tracks
    if not points:
        for route in gpx.routes:
            for pt in route.points:
                points.append(
                    {
                        "lat": pt.latitude,
                        "lon": pt.longitude,
                        "ele": pt.elevation,
                    }
                )

    return points
