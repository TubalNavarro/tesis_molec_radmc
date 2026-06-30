#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# ============================================================
# Configuración
# ============================================================

BASE_DIR = Path(".")   # cambia esto si corres el script desde otro lugar
OUTPUT_DIR = BASE_DIR / "pv_collections_G328_Test"

PV_FILES = ["pv.fits", "pv_residuals.fits"]
CUBE_FILE = "csub_convolved_noise.fits"

GROUPS = {
    "Rdisc": "model_Ulrich_rdisc=*",
    "Arho0": "model_Ulrich_dens_Arho0=*",
    "BT": "model_Ulrich_BT=*",
    "Incl": "model_Ulrich_i=*",
    "T10": "model_Ulrich_T10=*",
    "MStar": "model_Ulrich_MStar=*",
    "MRate": "model_Ulrich_Mrate=*",
    "test": "model_Ulrich_G328_Test",
    "Diagnostic": "model_Ulrich_Diagnostic*"
}


# ============================================================
# Funciones generales
# ============================================================

def safe_name(folder_name, fits_name):
    """
    Convierte, por ejemplo:
        folder_name = model_Ulrich_rdisc=150
        fits_name   = pv.fits

    en:
        model_Ulrich_rdisc_150_pv.fits
    """
    stem = Path(fits_name).stem

    clean_folder_name = (
        folder_name
        .replace("=", "_")
        .replace(" ", "_")
    )

    return f"{clean_folder_name}_{stem}.fits"


def safe_png_name(folder_name, suffix):
    """
    Nombre seguro para figuras PNG generadas a partir de una carpeta de modelo.
    """
    clean_folder_name = (
        folder_name
        .replace("=", "_")
        .replace(" ", "_")
    )

    return f"{clean_folder_name}_{suffix}.png"


# ============================================================
# Copiar PVs y convertirlos a PNG
# ============================================================

def copy_pv_files(base_dir=BASE_DIR, output_dir=OUTPUT_DIR):
    """
    Copia pv.fits y pv_residuals.fits a carpetas separadas según el grupo.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []

    for group_name, pattern in GROUPS.items():

        group_dir = output_dir / group_name
        group_pv_dir = group_dir / "fits"
        group_png_dir = group_dir / "png"

        group_pv_dir.mkdir(parents=True, exist_ok=True)
        group_png_dir.mkdir(parents=True, exist_ok=True)

        folders = sorted(base_dir.glob(pattern))

        print(f"\nGrupo: {group_name}")
        print(f"Patrón: {pattern}")
        print(f"Carpetas encontradas: {len(folders)}")

        for folder in folders:

            if not folder.is_dir():
                continue

            for fits_file in PV_FILES:

                src = folder / fits_file

                if not src.exists():
                    print(f"  No existe: {src}")
                    continue

                dst_name = safe_name(folder.name, fits_file)
                dst = group_pv_dir / dst_name

                shutil.copy2(src, dst)
                copied_files.append(dst)

                print(f"  Copiado: {src} -> {dst}")

    return copied_files


def fits_to_png(fits_path, png_path=None, cmap="inferno", percentile_clip=(1, 99)):
    """
    Convierte un archivo FITS 2D a PNG.
    """

    fits_path = Path(fits_path)

    if png_path is None:
        png_path = fits_path.with_suffix(".png")
    else:
        png_path = Path(png_path)

    with fits.open(fits_path) as hdul:
        data = hdul[0].data

    data = np.squeeze(data)

    if data.ndim != 2:
        print(f"  Saltando {fits_path}, no es 2D después de squeeze. Shape = {data.shape}")
        return None

    finite = np.isfinite(data)

    if not np.any(finite):
        print(f"  Saltando {fits_path}, no tiene valores finitos.")
        return None

    vmin, vmax = np.nanpercentile(data[finite], percentile_clip)

    plt.figure(figsize=(7, 5))
    plt.imshow(
        data,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(label="Intensidad")
    plt.xlabel("Pixel")
    plt.ylabel("Pixel")
    plt.title(fits_path.stem)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"  PNG guardado: {png_path}")

    return png_path


def convert_all_copied_fits_to_png(output_dir=OUTPUT_DIR):
    """
    Convierte todos los FITS copiados dentro de pv_collections a PNG.
    Guarda los PNG en la carpeta png correspondiente de cada grupo.
    """

    output_dir = Path(output_dir)

    fits_files = sorted(output_dir.glob("*/fits/*.fits"))

    print(f"\nConvirtiendo FITS a PNG...")
    print(f"Archivos FITS encontrados: {len(fits_files)}")

    png_files = []

    for fits_path in fits_files:

        group_dir = fits_path.parents[1]
        png_dir = group_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)

        png_path = png_dir / fits_path.with_suffix(".png").name

        result = fits_to_png(fits_path, png_path=png_path)

        if result is not None:
            png_files.append(result)

    return png_files


# ============================================================
# Espectro 1D colapsado espacialmente
# ============================================================

def find_spectral_fits_axis(header):
    """
    Encuentra el eje espectral en el header FITS.

    Regresa el número de eje FITS, es decir, 1 para CTYPE1,
    2 para CTYPE2, etc.
    """

    naxis = int(header.get("NAXIS", 0))

    spectral_keywords = (
        "VRAD", "VOPT", "VELO", "FELO",
        "FREQ", "WAVE", "AWAV", "ENER"
    )

    for fits_axis in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{fits_axis}", "")).upper()

        if any(key in ctype for key in spectral_keywords):
            return fits_axis

    raise ValueError(
        "No pude identificar el eje espectral. "
        "Revisa CTYPE1, CTYPE2, CTYPE3, etc. en el header."
    )


def get_linear_axis_values(header, fits_axis, n_pix):
    """
    Calcula los valores de un eje lineal FITS usando CRVAL, CRPIX y CDELT.

    FITS usa pixeles 1-indexados:
        coord = CRVAL + (pixel - CRPIX) * CDELT
    """

    crval = float(header.get(f"CRVAL{fits_axis}", 0.0))
    crpix = float(header.get(f"CRPIX{fits_axis}", 1.0))
    cdelt = float(header.get(f"CDELT{fits_axis}", 1.0))

    pix = np.arange(n_pix, dtype=float) + 1.0
    values = crval + (pix - crpix) * cdelt

    return values


def spectral_axis_to_velocity_kms(header, fits_axis, n_pix):
    """
    Construye el eje de velocidad en km/s.

    Si el eje ya es de velocidad, convierte m/s -> km/s cuando haga falta.
    Si el eje es de frecuencia, intenta convertir a velocidad radial de radio
    usando RESTFRQ o RESTFREQ:

        v = c * (nu_rest - nu) / nu_rest

    En ese caso, v queda en km/s.
    """

    ctype = str(header.get(f"CTYPE{fits_axis}", "")).upper()
    cunit = str(header.get(f"CUNIT{fits_axis}", "")).lower().replace(" ", "")

    axis_values = get_linear_axis_values(header, fits_axis, n_pix)

    # Caso 1: el eje ya está en velocidad.
    if any(key in ctype for key in ("VRAD", "VOPT", "VELO", "FELO")):

        if cunit in ("m/s", "ms-1", "m.s-1", "meter/s", "meters/s"):
            velocity_kms = axis_values / 1000.0
        elif cunit in ("km/s", "kms-1", "km.s-1", "kilometer/s", "kilometers/s"):
            velocity_kms = axis_values
        else:
            # Muchos cubos guardan velocidad en m/s aunque CUNIT no sea claro.
            # Si los valores son muy grandes, asumimos m/s.
            if np.nanmax(np.abs(axis_values)) > 1.0e4:
                velocity_kms = axis_values / 1000.0
            else:
                velocity_kms = axis_values

        return velocity_kms

    # Caso 2: eje en frecuencia. Se convierte a velocidad de radio.
    if "FREQ" in ctype:

        freq = axis_values.copy()

        # Convertir frecuencia a Hz si hace falta.
        if cunit in ("ghz",):
            freq_hz = freq * 1.0e9
        elif cunit in ("mhz",):
            freq_hz = freq * 1.0e6
        elif cunit in ("khz",):
            freq_hz = freq * 1.0e3
        else:
            freq_hz = freq

        restfreq = header.get("RESTFRQ", header.get("RESTFREQ", None))

        if restfreq is None:
            raise ValueError(
                "El eje espectral está en frecuencia, pero no encontré RESTFRQ/RESTFREQ "
                "para convertir a velocidad."
            )

        restfreq = float(restfreq)
        c_kms = 299792.458

        velocity_kms = c_kms * (restfreq - freq_hz) / restfreq

        return velocity_kms

    raise ValueError(
        f"El eje espectral CTYPE{fits_axis}={ctype!r} no parece ser velocidad ni frecuencia."
    )


def spatially_averaged_spectrum(cube_path):
    """
    Lee un cubo espectral y calcula un espectro 1D promediado espacialmente.

    Para cada canal espectral promedia todos los pixeles espaciales finitos:

        spectrum[v] = mean_x,y I(v, y, x)

    Esto conserva la unidad original del cubo. Por ejemplo, si el cubo está en
    Jy/beam, el espectro también queda en Jy/beam.

    Nota importante:
    - Una suma espacial directa daría Jy/beam * pixel.
    - Una integral de flujo físico daría Jy y requeriría multiplicar por
      area_pixel / area_beam.
    - Aquí se usa promedio espacial porque quieres mantener Jy/beam.

    Returns
    -------
    velocity_kms : ndarray
        Eje de velocidad en km/s.

    spectrum : ndarray
        Intensidad promedio en cada canal, con la misma unidad que el cubo.

    bunit : str
        Unidad original del cubo, por ejemplo Jy/beam.
    """

    cube_path = Path(cube_path)

    with fits.open(cube_path) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header

    if data.ndim < 3:
        raise ValueError(f"{cube_path} no parece ser un cubo 3D. Shape = {data.shape}")

    fits_spectral_axis = find_spectral_fits_axis(header)

    # Relación entre ejes FITS y ejes numpy:
    # FITS:  axis 1, axis 2, axis 3
    # numpy: data[..., axis 2, axis 1]
    numpy_spectral_axis = data.ndim - fits_spectral_axis

    if numpy_spectral_axis < 0 or numpy_spectral_axis >= data.ndim:
        raise ValueError(
            f"No pude mapear el eje FITS {fits_spectral_axis} "
            f"a un eje numpy válido para shape={data.shape}."
        )

    n_chan = data.shape[numpy_spectral_axis]
    velocity_kms = spectral_axis_to_velocity_kms(header, fits_spectral_axis, n_chan)

    spatial_axes = tuple(ax for ax in range(data.ndim) if ax != numpy_spectral_axis)

    # Promedio espacial, no suma. Así se conserva BUNIT = Jy/beam.
    with np.errstate(invalid="ignore"):
        spectrum = np.nanmean(data, axis=spatial_axes)

    # Ordenar por velocidad para que la curva quede limpia aunque CDELT sea negativo.
    order = np.argsort(velocity_kms)
    velocity_kms = velocity_kms[order]
    spectrum = spectrum[order]

    bunit = str(header.get("BUNIT", "Jy/beam"))

    return velocity_kms, spectrum, bunit


def plot_spatially_averaged_spectrum(
    cube_path,
    png_path=None,
    title=None,
    xlim=None,
    ylim=None,
    save_txt=False,
):
    """
    Grafica velocidad vs intensidad promedio espacial.

    Parameters
    ----------
    cube_path : str or Path
        Cubo espectral, por ejemplo csub_convolved_noise.fits.

    png_path : str or Path, optional
        Ruta de salida de la figura PNG.

    title : str, optional
        Título de la figura.

    xlim : tuple, optional
        Rango de velocidad en km/s, por ejemplo (-60, -30).

    ylim : tuple, optional
        Rango vertical.

    save_txt : bool
        Si True, también guarda un .dat con dos columnas:
        velocidad_km_s, intensidad_promedio.
    """

    cube_path = Path(cube_path)

    if png_path is None:
        png_path = cube_path.with_name(cube_path.stem + "_spectrum.png")
    else:
        png_path = Path(png_path)

    velocity_kms, spectrum, bunit = spatially_averaged_spectrum(cube_path)

    plt.figure(figsize=(7, 4.5))
    plt.plot(velocity_kms, spectrum, lw=1.5)
    plt.axhline(0.0, ls="--", lw=0.8)

    plt.xlabel("Velocidad [km/s]")
    plt.ylabel(f"Intensidad promedio espacial [{bunit}]")

    if title is None:
        title = cube_path.parent.name

    plt.title(title)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"  Espectro guardado: {png_path}")

    if save_txt:
        txt_path = png_path.with_suffix(".dat")
        np.savetxt(
            txt_path,
            np.column_stack([velocity_kms, spectrum]),
            header="velocity_km_s  spatially_averaged_intensity",
        )
        print(f"  Datos del espectro guardados: {txt_path}")

    return png_path


def plot_all_spatially_averaged_spectra(
    base_dir=BASE_DIR,
    output_dir=OUTPUT_DIR,
    cube_file=CUBE_FILE,
    xlim=None,
    ylim=None,
    save_txt=False,
):
    """
    Busca el cubo csub_convolved_noise.fits en todas las carpetas de modelo,
    promedia espacialmente cada cubo y guarda una gráfica 1D velocidad-intensidad.

    Las figuras se guardan en:
        pv_collections_G328_Test/<grupo>/spectra_png/
    """

    base_dir = Path(base_dir)
    output_dir = Path(output_dir)

    spectrum_pngs = []

    print("\nGenerando espectros 1D promediados espacialmente...")

    for group_name, pattern in GROUPS.items():

        group_dir = output_dir / group_name
        spectra_dir = group_dir / "spectra_png"
        spectra_dir.mkdir(parents=True, exist_ok=True)

        folders = sorted(base_dir.glob(pattern))

        print(f"\nGrupo: {group_name}")
        print(f"Patrón: {pattern}")
        print(f"Carpetas encontradas: {len(folders)}")

        for folder in folders:

            if not folder.is_dir():
                continue

            cube_path = folder / cube_file

            if not cube_path.exists():
                print(f"  No existe: {cube_path}")
                continue

            png_name = safe_png_name(folder.name, "spectrum_1d")
            png_path = spectra_dir / png_name

            try:
                result = plot_spatially_averaged_spectrum(
                    cube_path,
                    png_path=png_path,
                    title=folder.name,
                    xlim=xlim,
                    ylim=ylim,
                    save_txt=save_txt,
                )

                spectrum_pngs.append(result)

            except Exception as err:
                print(f"  Error con {cube_path}: {err}")

    return spectrum_pngs


# ============================================================
# Ejecución principal
# ============================================================

if __name__ == "__main__":

    copied = copy_pv_files()
    pngs = convert_all_copied_fits_to_png()

    # Esto NO hace momento 0.
    # Promedia espacialmente el cubo para obtener un espectro 1D:
    # velocidad vs intensidad promedio en toda la región espacial.
    # Así se conserva la unidad Jy/beam.
    spectra_pngs = plot_all_spatially_averaged_spectra(
        xlim=None,      # ejemplo: xlim=(-60, -30)
        ylim=None,
        save_txt=False, # cambia a True si también quieres guardar los datos .dat
    )

    print("\nResumen:")
    print(f"FITS copiados: {len(copied)}")
    print(f"PNG de PV generados: {len(pngs)}")
    print(f"Espectros 1D generados: {len(spectra_pngs)}")
    print(f"Salida en: {OUTPUT_DIR}")
