
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import astropy.units as u
from astropy.io import fits



BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "pv_collections_G328_Test"

PV_FILES = ["pv.fits", "pv_residuals.fits"]

# Frecuencia de reposo de la línea.
REST_FREQUENCY = 335.582017 * u.GHz

GROUPS = {
    "Rdisc": "model_Ulrich_rdisc=*",
    "Arho0": "model_Ulrich_dens_Arho0=*",
    "BT": "model_Ulrich_BT=*",
    "Incl": "model_Ulrich_i=*",
    "test": "model_Ulrich_test",
    "MStar": "model_Ulrich_MStar=*",
    "MRate": "model_Ulrich_Mrate=*",
    "T10": "model_Ulrich_T10=*",
    "Exp": "model_Ulrich_exp=*",
    "Abund": "model_Ulrich_abund=*",
    "Diagnostic": "model_Ulrich_nodisk_Diagnostic*",
    "nodisc_Rdisc": "model_Ulrich_nodisk_rdisc=*",
    "nodisc_Arho0": "model_Ulrich_nodisk_dens_Arho0=*",
    "nodisc_BT": "model_Ulrich_nodisk_BT=*",
    "nodisc_Incl": "model_Ulrich_nodisk_i=*",
    "nodisc_test": "model_Ulrich_nodisk_G328_test",
    "nodisc_MStar": "model_Ulrich_nodisk_MStar=*",
    "nodisc_MRate": "model_Ulrich_nodisk_Mrate=*",
    "nodisc_T10": "model_Ulrich_nodisk_T10=*",
    "nodisc_Diagnostic": "model_Ulrich_nodisk_Diagnostic*",
    "nodisc_Exp": "model_Ulrich_nodisk_exp=*",
    "nodisc_abund": "model_Ulrich_abund=*",
}


# Funciones auxiliares

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


def get_rest_frequency(header, default_rest_frequency=None):

    possible_keys = [
        "RESTFRQ",
        "RESTFREQ",
        "RESTFREQU",
        "REST_FREQ",
    ]

    for key in possible_keys:
        if key in header:
            return header[key] * u.Hz

    if default_rest_frequency is not None:
        return default_rest_frequency

    raise ValueError(
        "No encontré RESTFRQ/RESTFREQ en el header. "
        "Define REST_FREQUENCY manualmente."
    )


def get_axis_centers_from_header(
    header,
    axis_number,
    n_pix,
    output_unit=None,
    rest_frequency=None,
):


    crpix = header.get(f"CRPIX{axis_number}", 1.0)
    crval = header.get(f"CRVAL{axis_number}", 0.0)
    cdelt = header.get(f"CDELT{axis_number}", 1.0)
    cunit = header.get(f"CUNIT{axis_number}", "")

   
    pix_centers = np.arange(n_pix) + 1

    world_centers = crval + (pix_centers - crpix) * cdelt

    if cunit not in ["", None]:

        try:
            unit = u.Unit(cunit)
            world_centers = world_centers * unit

            if output_unit is not None:

                try:
                    world_centers = world_centers.to(output_unit)

                except u.UnitConversionError:
                    if unit.is_equivalent(u.Hz) and output_unit.is_equivalent(u.km / u.s):

                        restfreq = get_rest_frequency(
                            header,
                            default_rest_frequency=rest_frequency,
                        )

                        world_centers = world_centers.to(
                            output_unit,
                            equivalencies=u.doppler_radio(restfreq),
                        )

                    else:
                        print(
                            f"  Advertencia: no pude convertir "
                            f"CUNIT{axis_number}='{cunit}' a {output_unit}."
                        )
                        print("  Se usarán las unidades originales del header.")

            # Quitar unidades para evitar errores con arrays de float
            world_centers = world_centers.value

        except Exception as e:
            print(
                f"  Advertencia: problema leyendo CUNIT{axis_number}='{cunit}'."
            )
            print(f"  Se usarán valores numéricos sin convertir. Error: {e}")

            world_centers = np.asarray(
                crval + (pix_centers - crpix) * cdelt,
                dtype=float,
            )

    else:
        world_centers = np.asarray(world_centers, dtype=float)

    return world_centers


def centers_to_edges(centers):
    
    #Convierte centros de pixeles a bordes para usar con imshow(extent=...).

    centers = np.asarray(centers, dtype=float)
    n_pix = len(centers)

    if n_pix == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])

    edges = np.empty(n_pix + 1)

    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])

    first_step = centers[1] - centers[0]
    last_step = centers[-1] - centers[-2]

    edges[0] = centers[0] - 0.5 * first_step
    edges[-1] = centers[-1] + 0.5 * last_step

    return edges


def get_axis_edges_from_header(
    header,
    axis_number,
    n_pix,
    output_unit=None,
    rest_frequency=None,
):
    
    #Construye los bordes del eje físico para usar con imshow(extent=...).
    centers = get_axis_centers_from_header(
        header=header,
        axis_number=axis_number,
        n_pix=n_pix,
        output_unit=output_unit,
        rest_frequency=rest_frequency,
    )

    return centers_to_edges(centers)


def recenter_edges_to_middle(edges):
    
    #Re-centra un eje para que el centro geométrico del mapa sea 0.

    edges = np.asarray(edges, dtype=float)

    center = 0.5 * (edges[0] + edges[-1])

    return edges - center


def get_colorbar_label(header, is_residual=False):
    
    #Construye la etiqueta de la barra de color usando BUNIT si existe.
  

    bunit = header.get("BUNIT", "").strip()

    if is_residual:
        label = "Residual"
    else:
        label = "Intensidad"

    if bunit != "":
        label += f" [{bunit}]"

    return label


# ============================================================
# Conversión FITS -> PNG
# ============================================================

def fits_to_png(
    fits_path,
    png_path=None,
    cmap="inferno",
    residual_cmap="RdBu_r",
    percentile_clip=(1, 99),
):

    fits_path = Path(fits_path)

    if png_path is None:
        png_path = fits_path.with_suffix(".png")
    else:
        png_path = Path(png_path)

    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    data = np.squeeze(data)

    if data.ndim != 2:
        print(
            f"  Saltando {fits_path}, no es 2D después de squeeze. "
            f"Shape = {data.shape}"
        )
        return None

    finite = np.isfinite(data)

    if not np.any(finite):
        print(f"  Saltando {fits_path}, no tiene valores finitos.")
        return None

    ny, nx = data.shape

    # ========================================================
    # Ejes físicos del diagrama PV
    # ========================================================


    x_edges = get_axis_edges_from_header(
        header=header,
        axis_number=1,
        n_pix=nx,
        output_unit=u.arcsec,
    )

    x_edges = recenter_edges_to_middle(x_edges)

    y_edges = get_axis_edges_from_header(
        header=header,
        axis_number=2,
        n_pix=ny,
        output_unit=u.km / u.s,
        rest_frequency=REST_FREQUENCY,
    )

    extent = [
        x_edges[0],
        x_edges[-1],
        y_edges[0],
        y_edges[-1],
    ]

    # Detecta si es un archivo de residuos
    is_residual = "residual" in fits_path.stem.lower()


    plt.figure(figsize=(7, 5))

    if is_residual:

        # Escala simétrica alrededor de cero
        max_abs = np.nanpercentile(
            np.abs(data[finite]),
            percentile_clip[1],
        )

        if max_abs == 0 or not np.isfinite(max_abs):
            max_abs = 1.0

        norm = TwoSlopeNorm(
            vmin=-max_abs,
            vcenter=0.0,
            vmax=max_abs,
        )

        im = plt.imshow(
            data,
            origin="lower",
            cmap=residual_cmap,
            aspect="auto",
            extent=extent,
            norm=norm,
        )

    else:

        vmin, vmax = np.nanpercentile(data[finite], percentile_clip)

        if vmin == vmax:
            vmin = np.nanmin(data[finite])
            vmax = np.nanmax(data[finite])

        im = plt.imshow(
            data,
            origin="lower",
            cmap=cmap,
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )

    cbar_label = get_colorbar_label(header, is_residual=is_residual)
    plt.colorbar(im, label=cbar_label)

    plt.xlabel("Offset [arcsec]")
    plt.ylabel(r"$v_\mathrm{rad}$ [km s$^{-1}$]")
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

        result = fits_to_png(
            fits_path=fits_path,
            png_path=png_path,
            cmap="inferno",
            residual_cmap="RdBu_r",
            percentile_clip=(1, 99),
        )

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