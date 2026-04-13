# ---------------------------------------------------------------------------
# HRDEM — High Resolution DEM (1 m, watershed tiles via NRCan FTP)
# ---------------------------------------------------------------------------
# COVERAGE WARNING: HRDEM is collected for flood-mapping and urban priorities,
# NOT general mountain terrain. Alpine areas such as Bow Summit (Icefields
# Parkway), Rogers Pass, and most backcountry terrain have NO coverage.
# A comprehensive search of all AB and BC UTM11 FTP directories confirmed zero
# tiles exist for the Bow Summit area (expected tile indices e_3_172/173).
# Use fetch_dem_mrdem() for consistent nationwide coverage at 30 m instead.
# fetch_dem_hrdem() remains useful for validation areas that happen to fall
# inside an HRDEM watershed footprint (river corridors, urban fringes).
#
# FTP root: https://ftp.maps.canada.ca/pub/elevation/dem_mne/
#               highresolution_hauteresolution/dtm_mnt/1m/{province}/{watershed}/utm{zone}/
# Tile convention: dtm_1m_utm{zone}_e_{e}_{n}.tif
#   left_easting   = (e + 50)  × 10 000 m
#   bottom_northing = (n + 400) × 10 000 m
# Native CRS: EPSG:2955 (NAD83(CSRS) / UTM Zone 11N), nodata = -32767.

_HRDEM_NODATA = -32767.0
_HRDEM_TILE_SIZE = 10_000  # metres per tile side (10 km × 10 km at 1 m resolution)
_HRDEM_E_OFFSET = 50  # left_easting   = (e_idx + 50)  × 10 000
_HRDEM_N_OFFSET = 400  # bottom_northing = (n_idx + 400) × 10 000


# def fetch_dem_hrdem(
#     min_lat: float,
#     min_lon: float,
#     max_lat: float,
#     max_lon: float,
#     tile_base_url: str,
# ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
#     """Fetch 1 m HRDEM tiles for the given WGS84 bbox from an NRCan FTP directory.

#     .. warning::
#         HRDEM covers flood-mapping and urban areas only — **not** alpine or
#         backcountry terrain. Most ATES validation areas will return a
#         ``FileNotFoundError``. Use :func:`fetch_dem_mrdem` for consistent
#         nationwide coverage. See module-level ``_HRDEM_*`` comments for details.

#     Args:
#         min_lat, min_lon, max_lat, max_lon: WGS84 bounding box.
#         tile_base_url: URL to the watershed UTM subdirectory, e.g.
#             ``"https://ftp.maps.canada.ca/.../1m/AB/Upper_Bow_River/utm11"``.

#     Returns:
#         ``(dem, latlon_bounds)`` — same format as :func:`fetch_dem_wcs`.

#     Raises:
#         FileNotFoundError: if no tiles exist for the bbox in ``tile_base_url``.
#     """
#     wgs84 = CRS.from_epsg(4326)

#     utm_zone = int(tile_base_url.rstrip("/").rsplit("utm", 1)[-1])
#     approx_utm_crs = CRS.from_epsg(32600 + utm_zone)

#     left_u, bottom_u, right_u, top_u = transform_bounds(
#         wgs84, approx_utm_crs, min_lon, min_lat, max_lon, max_lat
#     )

#     e_min = int(left_u   // _HRDEM_TILE_SIZE) - _HRDEM_E_OFFSET
#     e_max = int(right_u  // _HRDEM_TILE_SIZE) - _HRDEM_E_OFFSET
#     n_min = int(bottom_u // _HRDEM_TILE_SIZE) - _HRDEM_N_OFFSET
#     n_max = int(top_u    // _HRDEM_TILE_SIZE) - _HRDEM_N_OFFSET

#     base = tile_base_url.rstrip("/")
#     prefix = f"dtm_1m_utm{utm_zone}"

#     datasets = []
#     for e in range(e_min, e_max + 1):
#         for n in range(n_min, n_max + 1):
#             url = f"{base}/{prefix}_e_{e}_{n}.tif"
#             try:
#                 datasets.append(rasterio.open(url))
#             except rasterio.errors.RasterioIOError:
#                 pass  # tile absent from this watershed directory

#     if not datasets:
#         raise FileNotFoundError(
#             f"No HRDEM tiles found for bbox ({min_lat},{min_lon},{max_lat},{max_lon}) "
#             f"in {tile_base_url}. "
#             f"Expected tile range: e_{e_min}–{e_max}, n_{n_min}–{n_max}. "
#             f"Consider fetch_dem_mrdem() for areas without HRDEM coverage."
#         )

#     native_crs = datasets[0].crs
#     left_n, bottom_n, right_n, top_n = transform_bounds(
#         wgs84, native_crs, min_lon, min_lat, max_lon, max_lat
#     )

#     mosaic, mosaic_transform = _rasterio_merge(
#         datasets,
#         bounds=(left_n, bottom_n, right_n, top_n),
#         nodata=_HRDEM_NODATA,
#     )
#     for ds in datasets:
#         ds.close()

#     dem = mosaic[0].astype(np.float32)
#     dem[dem == _HRDEM_NODATA] = np.nan

#     height, width = dem.shape
#     dst_transform, dst_width, dst_height = calculate_default_transform(
#         native_crs, wgs84, width, height,
#         left=mosaic_transform.c,
#         bottom=mosaic_transform.f + mosaic_transform.e * height,
#         right=mosaic_transform.c + mosaic_transform.a * width,
#         top=mosaic_transform.f,
#     )
#     dem_wgs84 = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
#     reproject(
#         source=dem,
#         destination=dem_wgs84,
#         src_transform=mosaic_transform,
#         src_crs=native_crs,
#         dst_transform=dst_transform,
#         dst_crs=wgs84,
#         resampling=Resampling.bilinear,
#         src_nodata=np.nan,
#         dst_nodata=np.nan,
#     )

#     left = dst_transform.c
#     top = dst_transform.f
#     right = left + dst_transform.a * dst_width
#     bottom = top + dst_transform.e * dst_height

#     return dem_wgs84, (bottom, left, top, right)
