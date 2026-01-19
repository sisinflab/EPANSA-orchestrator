import logging
import pathlib
from typing import Optional, Tuple, Dict, Any
import requests
from PIL import Image, ExifTags


def _get_exif_dict(img_path: pathlib.Path):
    """Returns the raw EXIF using Pillow."""
    with Image.open(img_path) as im:
        return im.getexif() or Image.Exif()


def _extract_gps_info(exif):
    """Extracts GPS info from EXIF dict."""
    if not exif:
        return None

    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if gps_ifd:
        return {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
    else:
        return None


def _rational_to_float(x):
    """Converts a Pillow rational value to float."""
    try:
        return float(x[0]) / float(x[1])
    except Exception:
        try:
            return float(x)
        except Exception:
            return None


def _dms_to_decimal(dms, ref) -> Optional[float]:
    """Converts DMS to decimal degrees."""
    if not dms or len(dms) != 3 or ref not in ("N", "S", "E", "W"):
        return None
    try:
        deg = _rational_to_float(dms[0])
        minute = _rational_to_float(dms[1])
        sec = _rational_to_float(dms[2])
        if None in (deg, minute, sec):
            return None

        dec = deg + (minute / 60.0) + (sec / 3600.0)
        if ref in ("S", "W"):
            dec = -dec
        return dec
    except Exception:
        return None


def _gps_to_decimal(gps: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Estrae lat/long in decimali dal dict GPS."""
    lat_dms = gps.get("GPSLatitude")
    lat_ref = gps.get("GPSLatitudeRef", "N")
    lon_dms = gps.get("GPSLongitude")
    lon_ref = gps.get("GPSLongitudeRef", "E")

    lat = _dms_to_decimal(lat_dms, lat_ref)
    lon = _dms_to_decimal(lon_dms, lon_ref)
    if lat is None or lon is None:
        return None
    return lat, lon


def reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Calls Nominatim to get a textual description of the location."""
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "jsonv2",
        "lat": f"{lat:.7f}",
        "lon": f"{lon:.7f}",
        "zoom": 16,
        "addressdetails": 1,
        "extratags": 0,
    }
    headers = {"User-Agent": None, "Accept": "application/json"}
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning(f"Reverse geocoding FAILED: {e}")
        return None


def analyze_image(img_path: pathlib.Path) -> Optional[str]:
    """Analyzes the image and returns a location description."""
    logging.info("Getting image location...")
    if not img_path.exists() or not img_path.is_file():
        raise FileNotFoundError(f"File non trovato: {img_path}")

    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        if img_path.suffix.lower() == ".png":
            return None
        else:
            raise ValueError(f"Image extension not supported: {img_path.suffix}")

    exif = _get_exif_dict(img_path)
    gps = _extract_gps_info(exif)

    if gps:
        coords = _gps_to_decimal(gps)
        if coords:
            lat, lon = coords
            r = reverse_geocode(lat, lon)
            if r:
                location_desc = [
                    r.get("address").get("road", None),
                    r.get("address").get("quarter", None),
                    r.get("address").get("city", None),
                    r.get("address").get("country", None),
                ]
                return ", ".join([x for x in location_desc if x])

    else:
        logging.error("FAILED in getting image location")
        return None
