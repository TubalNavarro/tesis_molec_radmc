from astropy import units as u
from astropy.coordinates import SkyCoord
from spectral_cube import SpectralCube
from pvextractor import Path, extract_pv_slice
from astropy.io import fits
import numpy as np


def fix_pv_header_arcsec_and_centered_freq(
    pv,
    restfreq_GHz=233.7956660,
):
    """
    Corrige el header de un diagrama PV.

    Eje 1:
        Offset espacial en arcsec, con offset = 0 en el centro.

    Eje 2:
        Se mantiene en frecuencia, pero se reescribe el header para que
        CRVAL2 sea restfreq_GHz. Esto NO cambia los datos, solo cambia
        la referencia WCS del eje espectral.
    """

    header = pv.header

    # ========================================================
    # EJE ESPACIAL: convertir a arcsec y centrar offset = 0
    # ========================================================

    nx = header["NAXIS1"]

    cunit1 = header.get("CUNIT1", "").strip()

    if cunit1 == "":
        # pvextractor normalmente deja el eje espacial en grados
        cunit1 = "deg"

    try:
        factor_to_arcsec = u.Unit(cunit1).to(u.arcsec)
    except Exception:
        print(f"No pude interpretar CUNIT1 = {cunit1}. Asumo grados.")
        factor_to_arcsec = u.deg.to(u.arcsec)

    # Convertir escala espacial a arcsec/pixel
    if "CDELT1" in header:
        header["CDELT1"] = header["CDELT1"] * factor_to_arcsec

    # Por si el header usa matriz CD
    if "CD1_1" in header:
        header["CD1_1"] = header["CD1_1"] * factor_to_arcsec

    # Centrar el cero espacial
    header["CRVAL1"] = 0.0
    header["CRPIX1"] = (nx + 1) / 2.0
    header["CUNIT1"] = "arcsec"
    header["CTYPE1"] = "OFFSET"

    # ========================================================
    # EJE ESPECTRAL: mantener frecuencia, pero recentrar
    # ========================================================

    restfreq = restfreq_GHz * u.GHz

    cunit2 = header.get("CUNIT2", "").strip()

    if cunit2 == "":
        # Si no dice unidad, asumimos Hz
        cunit2 = "Hz"
        header["CUNIT2"] = "Hz"

    try:
        restfreq_in_header_unit = restfreq.to_value(u.Unit(cunit2))
    except Exception:
        print(f"No pude interpretar CUNIT2 = {cunit2}. Asumo Hz.")
        cunit2 = "Hz"
        header["CUNIT2"] = "Hz"
        restfreq_in_header_unit = restfreq.to_value(u.Hz)

    old_crval2 = header["CRVAL2"]
    old_crpix2 = header["CRPIX2"]
    cdelt2 = header["CDELT2"]

    # Fórmula FITS:
    # freq = CRVAL2 + (pixel - CRPIX2) * CDELT2
    #
    # Queremos conservar las frecuencias físicas de cada pixel,
    # pero escribir CRVAL2 = restfreq.
    #
    # Entonces:
    # CRPIX2_new = CRPIX2_old + (restfreq - CRVAL2_old) / CDELT2

    new_crpix2 = old_crpix2 + (restfreq_in_header_unit - old_crval2) / cdelt2

    header["CRVAL2"] = restfreq_in_header_unit
    header["CRPIX2"] = new_crpix2
    header["CTYPE2"] = "FREQ"

    # Guardar frecuencia de reposo también como metadato
    header["RESTFRQ"] = restfreq.to_value(u.Hz)
    header["RESTFREQ"] = restfreq.to_value(u.Hz)

    header["HISTORY"] = "Spatial axis converted to arcsec and recentered."
    header["HISTORY"] = f"Spectral frequency axis recentered at {restfreq_GHz} GHz."

    return pv


def make_pv_diagram(
    cube_file,
    output_file,
    center_coord,
    pa_deg,
    length_arcsec,
    width_arcsec=None,
    spacing_arcsec=0.01,
    restfreq_GHz=233.7956660,
):
    """
    Extrae un diagrama PV.

    El eje espacial queda en arcsec, con offset = 0 en el centro.
    El eje espectral queda en frecuencia, pero con CRVAL2 referido a
    restfreq_GHz.
    """

    cube = SpectralCube.read(cube_file)

    half_length = 0.5 * length_arcsec * u.arcsec
    pa = pa_deg * u.deg

    start = center_coord.directional_offset_by(
        pa + 180*u.deg,
        half_length
    )

    end = center_coord.directional_offset_by(
        pa,
        half_length
    )

    coords = SkyCoord(
        ra=[start.ra, end.ra],
        dec=[start.dec, end.dec],
        frame=center_coord.frame
    )

    if width_arcsec is None:
        path = Path(coords)
    else:
        path = Path(coords, width=width_arcsec*u.arcsec)

    pv = extract_pv_slice(
        cube,
        path,
        spacing=spacing_arcsec*u.arcsec
    )

    pv = fix_pv_header_arcsec_and_centered_freq(
        pv,
        restfreq_GHz=restfreq_GHz
    )

    pv.writeto(output_file, overwrite=True)

    print(f"PV guardado en: {output_file}")
    print(f"Centro del corte: {center_coord.to_string('hmsdms')}")
    print(f"Inicio del corte: {start.to_string('hmsdms')}")
    print(f"Final del corte:  {end.to_string('hmsdms')}")
    print(f"PA = {pa_deg} deg")
    print(f"Longitud = {length_arcsec} arcsec")
    print(f"Ancho = {width_arcsec} arcsec")
    print(f"Spacing = {spacing_arcsec} arcsec")
    print(f"Frecuencia de referencia = {restfreq_GHz} GHz")
    print(f"Shape del PV = {pv.data.shape}")

    print("")
    print("Header eje espacial:")
    print(f"CTYPE1 = {pv.header.get('CTYPE1')}")
    print(f"CUNIT1 = {pv.header.get('CUNIT1')}")
    print(f"CRPIX1 = {pv.header.get('CRPIX1')}")
    print(f"CRVAL1 = {pv.header.get('CRVAL1')}")
    print(f"CDELT1 = {pv.header.get('CDELT1')}")

    print("")
    print("Header eje espectral:")
    print(f"CTYPE2 = {pv.header.get('CTYPE2')}")
    print(f"CUNIT2 = {pv.header.get('CUNIT2')}")
    print(f"CRPIX2 = {pv.header.get('CRPIX2')}")
    print(f"CRVAL2 = {pv.header.get('CRVAL2')}")
    print(f"CDELT2 = {pv.header.get('CDELT2')}")
    print(f"RESTFRQ = {pv.header.get('RESTFRQ')}")

    return pv


def compute_residuals(
    pv1_file,
    pv2_file,
    output_file,
    reference="pv1",
    floor=None,
):
    """
    Calcula residuos relativos entre dos PV:

        residual = (PV1 - PV2) / reference

    reference puede ser:
        'pv1'
        'pv2'
        'mean'
    """

    with fits.open(pv1_file) as hdul1, fits.open(pv2_file) as hdul2:
        data1 = hdul1[0].data.astype(float)
        data2 = hdul2[0].data.astype(float)
        header = hdul1[0].header.copy()

    if data1.shape != data2.shape:
        raise ValueError(
            f"Los PV no tienen la misma forma: "
            f"{pv1_file} tiene {data1.shape}, "
            f"{pv2_file} tiene {data2.shape}"
        )

    if reference == "pv1":
        denominator = data1
    elif reference == "pv2":
        denominator = data2
    elif reference == "mean":
        denominator = 0.5 * (data1 + data2)
    else:
        raise ValueError("reference debe ser 'pv1', 'pv2' o 'mean'.")

    residual = np.full_like(data1, np.nan, dtype=float)

    if floor is None:
        valid = denominator != 0
    else:
        valid = np.abs(denominator) >= floor

    residual[valid] = (data1[valid] - data2[valid]) / denominator[valid]

    header["BUNIT"] = "relative"
    header["HISTORY"] = f"Relative residuals from {pv1_file} and {pv2_file}"
    header["COMMENT"] = "Residual = (PV1 - PV2) / reference"

    fits.writeto(output_file, residual, header, overwrite=True)

    print(f"Residuos relativos guardados en: {output_file}")
    print(f"Shape: {residual.shape}")
    print(f"Reference: {reference}")

    return residual


# ============================================================
# Parámetros del usuario
# ============================================================

cube_file = "/home/tubal/repo_tesis/cubes/DIHCA_cubes/shared_data/G335.78/G355.78+0.17_2_subcubefix_head.fits"

output_file = "/home/tubal/repo_tesis/tesis_molec_radmc/pv/pv_G355.78+0.17_2_freq.fits"

center = SkyCoord(
    "16h29m46.12974s",
    "-48d15m49.9512s",
    frame="icrs"
)

pa_deg = 65.0
length_arcsec = 0.991
width_arcsec = 0.03
spacing_arcsec = 0.01
restfreq_GHz = 233.7956660

pv = make_pv_diagram(
    cube_file=cube_file,
    output_file=output_file,
    center_coord=center,
    pa_deg=pa_deg,
    length_arcsec=length_arcsec,
    width_arcsec=width_arcsec,
    spacing_arcsec=spacing_arcsec,
    restfreq_GHz=restfreq_GHz,
)