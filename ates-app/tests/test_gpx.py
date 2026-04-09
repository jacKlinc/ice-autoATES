import io

import pytest

from ates.gpx import parse_gpx

FIXTURE_GPX = """\
<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <trk>
    <trkseg>
      <trkpt lat="51.6976" lon="-116.4934"><ele>2088</ele></trkpt>
      <trkpt lat="51.6990" lon="-116.4950"><ele>2150</ele></trkpt>
      <trkpt lat="51.7010" lon="-116.4970"><ele>2300</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
"""

FIXTURE_GPX_NO_ELE = """\
<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <trk>
    <trkseg>
      <trkpt lat="51.0" lon="-116.0"/>
      <trkpt lat="51.1" lon="-116.1"/>
    </trkseg>
  </trk>
</gpx>
"""

FIXTURE_GPX_ROUTE = """\
<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <rte>
    <rtept lat="52.0" lon="-115.0"><ele>1800</ele></rtept>
    <rtept lat="52.1" lon="-115.1"><ele>1900</ele></rtept>
  </rte>
</gpx>
"""


def test_parse_track_point_count():
    points = parse_gpx(FIXTURE_GPX.encode())
    assert len(points) == 3


def test_parse_track_coordinates():
    points = parse_gpx(FIXTURE_GPX.encode())
    assert points[0]["lat"] == pytest.approx(51.6976)
    assert points[0]["lon"] == pytest.approx(-116.4934)
    assert points[0]["ele"] == pytest.approx(2088)


def test_parse_track_last_point():
    points = parse_gpx(FIXTURE_GPX.encode())
    assert points[-1]["lat"] == pytest.approx(51.7010)
    assert points[-1]["ele"] == pytest.approx(2300)


def test_parse_missing_elevation():
    points = parse_gpx(FIXTURE_GPX_NO_ELE.encode())
    assert len(points) == 2
    assert points[0]["ele"] is None


def test_parse_route_fallback():
    """Routes (not tracks) should be parsed when no tracks are present."""
    points = parse_gpx(FIXTURE_GPX_ROUTE.encode())
    assert len(points) == 2
    assert points[0]["lat"] == pytest.approx(52.0)


def test_parse_accepts_bytes():
    points = parse_gpx(FIXTURE_GPX.encode())
    assert len(points) == 3


def test_parse_accepts_file_obj():
    points = parse_gpx(io.BytesIO(FIXTURE_GPX.encode()))
    assert len(points) == 3
