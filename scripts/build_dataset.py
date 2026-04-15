from pathlib import Path
import os
import sys
import multiprocessing as mp
import tempfile

import numpy as np
import psutil
import rasterio
from pyinstrument import Profiler
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoates_v2.PRA.PRA_AutoATES import PRA
from ates.dem import fetch_dem_mrdem, fetch_canopy_height_mrdem

# FlowPy directory — add to sys.path before importing flow_core / raster_io.
# We import these directly rather than going through main.py, which unconditionally
# imports Simulation.py → PyQt5 even in headless/batch mode.
_FLOWPY_DIR = Path(__file__).parent.parent / "autoates_v2" / "FlowPy_detrainment"

# Standard AutoATES v2 FlowPy parameters (from GUI defaults + Bow Summit inputpara.csv)
_FLOWPY_ALPHA = 25  # friction angle (degrees)
_FLOWPY_EXP = 8  # flow exponent


def _write_geotiff(
    path: Path, arr: np.ndarray, transform, crs, nodata=-9999, dtype="float32"
):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype=dtype,
        count=1,
        crs=crs,
        width=arr.shape[1],
        height=arr.shape[0],
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def _run_flowpy(dem_path: Path, release_path: Path, output_dir: Path, alpha: int, exp: int):
    """Call FlowPy's flow_core directly, bypassing main.py (which imports PyQt5)."""
    flowpy_str = str(_FLOWPY_DIR)
    if flowpy_str not in sys.path:
        sys.path.insert(0, flowpy_str)

    import raster_io as io
    import flow_core as fc

    dem, header = io.read_raster(str(dem_path))
    release, release_header = io.read_raster(str(release_path))

    if header["ncols"] != release_header["ncols"] or header["nrows"] != release_header["nrows"]:
        raise ValueError("DEM and release layer dimensions don't match")

    forest = np.zeros_like(dem)
    flux_threshold = 3e-4
    max_z = 8848

    available_memory = psutil.virtual_memory()[1]
    max_procs = max(1, int(available_memory / (dem.nbytes * 10)))

    n_splits = min(mp.cpu_count() * 4, max_procs)
    release_list = fc.split_release(release, release_header, n_splits)

    print(f"FlowPy: {len(release_list)} processes, alpha={alpha}, exp={exp}")
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(
            fc.calculation_effect,
            [[dem, header, forest, rp, alpha, exp, flux_threshold, max_z] for rp in release_list],
        )

    z_delta = np.zeros_like(dem)
    flux = np.zeros_like(dem)
    cell_counts = np.zeros_like(dem)
    z_delta_sum = np.zeros_like(dem)
    fp_ta = np.zeros_like(dem)
    fp_dis = np.ones_like(dem) * 10000

    for res in results:
        res = list(res)
        z_delta = np.maximum(z_delta, res[0])
        flux = np.maximum(flux, res[1])
        cell_counts += res[2]
        z_delta_sum += res[3]
        fp_ta = np.maximum(fp_ta, res[5])
        fp_dis = np.minimum(fp_dis, res[6])

    output_dir.mkdir(exist_ok=True)
    io.output_raster(str(dem_path), str(output_dir / "cell_counts.tif"), cell_counts)
    io.output_raster(str(dem_path), str(output_dir / "FP_travel_angle.tif"), fp_ta)
    io.output_raster(str(dem_path), str(output_dir / "flux.tif"), flux)

    return output_dir


def benchmark_one(kmz_path: Path, utm_resolution_m: float = 30.0):
    """Run the full feature generation pipeline for one KMZ area under pyinstrument."""
    import geopandas as gpd
    from rasterio.transform import from_bounds

    zones = gpd.read_file(kmz_path, driver="libkml", layer="Zones")
    utm_crs = zones.estimate_utm_crs()
    wgs84 = CRS.from_epsg(4326)
    minx, miny, maxx, maxy = zones.total_bounds

    # --- 1. Fetch MRDEM DTM ---
    dem_wgs84, (lat0, lon0, lat1, lon1) = fetch_dem_mrdem(miny, minx, maxy, maxx)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        wgs84,
        utm_crs,
        dem_wgs84.shape[1],
        dem_wgs84.shape[0],
        left=lon0,
        bottom=lat0,
        right=lon1,
        top=lat1,
        resolution=utm_resolution_m,
    )
    dem_utm = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        dem_wgs84,
        dem_utm,
        src_transform=from_bounds(
            lon0, lat0, lon1, lat1, dem_wgs84.shape[1], dem_wgs84.shape[0]
        ),
        src_crs=wgs84,
        dst_transform=dst_transform,
        dst_crs=utm_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # --- 2. Fetch canopy height ---
    canopy_wgs84, _ = fetch_canopy_height_mrdem(miny, minx, maxy, maxx)
    canopy_utm = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        canopy_wgs84,
        canopy_utm,
        src_transform=from_bounds(
            lon0, lat0, lon1, lat1, canopy_wgs84.shape[1], canopy_wgs84.shape[0]
        ),
        src_crs=wgs84,
        dst_transform=dst_transform,
        dst_crs=utm_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    with tempfile.TemporaryDirectory() as wd:
        wd = Path(wd)
        dem_path = wd / "dem.tif"
        canopy_path = wd / "canopy.tif"
        pra_path = wd / "pra_binary.tif"

        dem_disk = np.where(np.isfinite(dem_utm), dem_utm, -9999.0).astype(np.float32)
        _write_geotiff(dem_path, dem_disk, dst_transform, utm_crs, nodata=-9999.0)

        canopy_disk = np.where(np.isfinite(canopy_utm), canopy_utm, 0.0).astype(np.float32)
        _write_geotiff(canopy_path, canopy_disk, dst_transform, utm_crs, nodata=0.0)

        # --- 3. Run PRA ---
        # PRA writes outputs relative to cwd, so chdir into the temp dir first.
        # Use no_forest to avoid the bav/sen2cc forest-read bug in PRA_AutoATES.py.
        os.chdir(wd)
        PRA(
            "no_forest",
            str(dem_path),
            str(dem_path),
            radius=2,
            prob=0.5,
            winddir=0,
            windtol=180,
            pra_thd=0.15,
            sf=3,
        )
        (wd / "PRA" / "PRA_binary.tif").rename(pra_path)

        # --- 4. Run FlowPy ---
        print("DEM shape:", dem_utm.shape)
        res_dir = _run_flowpy(dem_path, pra_path, wd / "res", _FLOWPY_ALPHA, _FLOWPY_EXP)
        print("FlowPy outputs:", [p.name for p in res_dir.iterdir()])


if __name__ == "__main__":
    mp.set_start_method("spawn")

    kmz = (
        Path(__file__).parent.parent
        / "data"
        / "validation"
        / "brandywine"
        / "layers.kmz"
    )
    print(f"Benchmarking on: {kmz.parent.name}\n")

    profiler = Profiler()
    profiler.start()
    benchmark_one(kmz)
    profiler.stop()
    profiler.print()
