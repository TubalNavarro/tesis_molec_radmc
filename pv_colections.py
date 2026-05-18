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
OUTPUT_DIR = BASE_DIR / "pv_collections_Hamb"

PV_FILES = ["pv.fits", "pv_residuals.fits"]

GROUPS = {
    "Hamb_inclination": "model_Hamb_i=*",
    "Hamb_rdisc": "model_Hamb_rdisc=*",
    "Hamb_exponent": "model_Hamb_dens_exp=*",
}


# ============================================================
# Funciones
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

    Parameters
    ----------
    fits_path : str or Path
        Ruta al archivo .fits.

    png_path : str or Path, optional
        Ruta de salida .png. Si no se da, usa el mismo nombre del FITS.

    cmap : str
        Mapa de color de matplotlib.

    percentile_clip : tuple
        Percentiles para ajustar el contraste.
        Por defecto usa percentiles 1 y 99.
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
# Ejecución principal
# ============================================================

if __name__ == "__main__":

    copied = copy_pv_files()
    pngs = convert_all_copied_fits_to_png()

    print("\nResumen:")
    print(f"FITS copiados: {len(copied)}")
    print(f"PNG generados: {len(pngs)}")
    print(f"Salida en: {OUTPUT_DIR}")