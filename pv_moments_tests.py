from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm, SymLogNorm
import astropy.units as u
from astropy.io import fits


# ============================================================
# Configuracion
# ============================================================

BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "pv_collections_G328_Test"

PV_FILES = ["pv.fits", "pv_residuals.fits"]

# Cubo principal usado para generar los mapas de momentos.
# Se prueban en este orden para que el script siga funcionando si en alguna
# corrida guardaste el cubo con un nombre ligeramente distinto.
CUBE_CANDIDATES = [
    "csub_convolved_noise.fits",
    "csub_convolved.fits",
    "raw_radmc_csub.fits",
]

# Umbral fijo para el momento 1: 5 sigma = 
MOMENT1_THRESHOLD = 2.1e-2

# Frecuencia de reposo de la linea.
REST_FREQUENCY = 335.582017 * u.GHz

GROUPS = {
    "Arho0": "model_ulrich_dens_Arho0=*",
    "BT":    "model_ulrich_BT=*",
    "Incl":  "model_ulrich_incl=*",
    "p_exp": "model_ulrich_p=*",
    "test_G335_2_ulrich":  "model_ulrich_G335.78+0.17_2_test",
    "test_G335_2_hamburger":  "model_hamburgers_G335.78+0.17_2_test",
    "MStar": "model_ulrich_MStar=*",
    "MRate": "model_ulrich_MRate=*",
    "T10":   "model_ulrich_T10*",
    "Exp":   "model_ulrich_exp=*",
    "Abund": "model_ulrich_molec_abund=*",
    "Renv":  "model_ulrich_Renv=*",
    "Cavity":"model_ulrich_cavity_ang*",
    "Rdisc": "model_ulrich_Rdisc=*",
}


# ============================================================
# Funciones auxiliares
# ============================================================

def clean_folder_name(folder_name):
    return (
        folder_name
        .replace("=", "_")
        .replace(" ", "_")
    )


def safe_name(folder_name, fits_name):
    """
    Convierte, por ejemplo:

        folder_name = model_Ulrich_rdisc=150
        fits_name   = pv.fits

    en:

        model_Ulrich_rdisc_150_pv.fits
    """

    stem = Path(fits_name).stem
    return f"{clean_folder_name(folder_name)}_{stem}.fits"


def moment_name(folder_name, moment_number, extension="fits"):
    """Nombre seguro para un mapa de momentos."""
    return f"{clean_folder_name(folder_name)}_moment{moment_number}.{extension}"


def copy_pv_files(base_dir=BASE_DIR, output_dir=OUTPUT_DIR):
    """
    Copia pv.fits y pv_residuals.fits a carpetas separadas segun el grupo.
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
        print(f"Patron: {pattern}")
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
        "No encontre RESTFRQ/RESTFREQ en el header. "
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
                    if (
                        unit.is_equivalent(u.Hz)
                        and output_unit.is_equivalent(u.km / u.s)
                    ):

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
                        print("  Se usaran las unidades originales del header.")

            # Quitar unidades para evitar errores con arrays de float
            world_centers = world_centers.value

        except Exception as e:
            print(
                f"  Advertencia: problema leyendo CUNIT{axis_number}='{cunit}'."
            )
            print(f"  Se usaran valores numericos sin convertir. Error: {e}")

            world_centers = np.asarray(
                crval + (pix_centers - crpix) * cdelt,
                dtype=float,
            )

    else:
        world_centers = np.asarray(world_centers, dtype=float)

    return world_centers


def centers_to_edges(centers):
    """Convierte centros de pixeles a bordes para imshow/extents e integracion."""

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
    """Construye los bordes del eje fisico para usar con imshow(extent=...)."""

    centers = get_axis_centers_from_header(
        header=header,
        axis_number=axis_number,
        n_pix=n_pix,
        output_unit=output_unit,
        rest_frequency=rest_frequency,
    )

    return centers_to_edges(centers)


def recenter_edges_to_middle(edges):
    """Re-centra un eje para que el centro geometrico del mapa sea 0."""

    edges = np.asarray(edges, dtype=float)
    center = 0.5 * (edges[0] + edges[-1])
    return edges - center


def get_colorbar_label(header, is_residual=False):
    """Construye la etiqueta de la barra de color usando BUNIT si existe."""

    bunit = header.get("BUNIT", "").strip()

    if is_residual:
        label = "Residual"
    else:
        label = "Intensidad"

    if bunit != "":
        label += f" [{bunit}]"

    return label


def find_cube_file(folder, cube_candidates=CUBE_CANDIDATES):
    """Devuelve el primer cubo existente dentro de una carpeta de modelo."""

    for cube_name in cube_candidates:
        candidate = folder / cube_name
        if candidate.exists():
            return candidate

    return None


def make_spatial_2d_header(input_header, shape, bunit, moment_number):
    """
    Construye un header 2D conservando la informacion espacial de los ejes 1 y 2.

    Esta rutina evita dejar en el FITS de salida keywords del eje espectral.
    """

    ny, nx = shape
    out = fits.Header()

    out["NAXIS"] = 2
    out["NAXIS1"] = nx
    out["NAXIS2"] = ny

    # Keywords WCS basicas para los dos ejes espaciales.
    for axis in (1, 2):
        for key_root in (
            "CTYPE",
            "CUNIT",
            "CRPIX",
            "CRVAL",
            "CDELT",
            "CROTA",
        ):
            key = f"{key_root}{axis}"
            if key in input_header:
                out[key] = input_header[key]

    # Matrices CD/PC, si existen.
    for i in (1, 2):
        for j in (1, 2):
            for prefix in ("CD", "PC"):
                key = f"{prefix}{i}_{j}"
                if key in input_header:
                    out[key] = input_header[key]

    # Informacion util que no depende de dimensionalidad.
    for key in (
        "BMAJ",
        "BMIN",
        "BPA",
        "OBJECT",
        "TELESCOP",
        "INSTRUME",
        "EQUINOX",
        "RADESYS",
        "SPECSYS",
        "RESTFRQ",
        "RESTFREQ",
    ):
        if key in input_header:
            out[key] = input_header[key]

    out["BUNIT"] = bunit
    out["MOMENT"] = moment_number

    return out


# ============================================================
# Conversion PV FITS -> PNG
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
        data = np.squeeze(hdul[0].data)
        header = hdul[0].header

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

    # Enmascara NaN e infinitos para aplicarles el color "bad".
    masked_data = np.ma.masked_invalid(data)

    ny, nx = data.shape

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

    is_residual = "residual" in fits_path.stem.lower()

    # Se copia el mapa para no modificar globalmente el cmap original.
    selected_cmap = residual_cmap if is_residual else cmap
    plot_cmap = plt.get_cmap(selected_cmap).copy()
    plot_cmap.set_bad(color="black", alpha=1.0)

    plt.figure(figsize=(7, 5))

    if is_residual:
        norm = TwoSlopeNorm(
            vmin=-100.0,
            vcenter=0.0,
            vmax=100.0,
        )

        im = plt.imshow(
            masked_data,
            origin="lower",
            cmap=plot_cmap,
            aspect="auto",
            extent=extent,
            norm=norm,
        )

    else:
        vmin, vmax = np.nanpercentile(
            data[finite],
            percentile_clip,
        )

        if vmin == vmax:
            vmin = np.nanmin(data[finite])
            vmax = np.nanmax(data[finite])

        im = plt.imshow(
            masked_data,
            origin="lower",
            cmap=plot_cmap,
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )

    cbar_label = get_colorbar_label(
        header,
        is_residual=is_residual,
    )

    if is_residual:
        cbar = plt.colorbar(
            im,
            label=cbar_label,
            ticks=np.arange(-100, 101, 20),
            extend="both",
        )
    else:
        cbar = plt.colorbar(
            im,
            label=cbar_label,
        )
    

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

    print("\nConvirtiendo FITS PV a PNG...")
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
# Mapas de momentos
# ============================================================

def get_cube_velocity_axis(
    header,
    n_channels,
    spectral_axis_number=3,
    rest_frequency=REST_FREQUENCY,
):
    """
    Lee el eje espectral del cubo y lo convierte a km/s.

    Se define v-vsys=0 en el punto medio del cubo. Esto usa directamente la
    simetria espectral indicada para estos modelos y no necesita leer vsys de
    ningun archivo externo.

    Returns
    -------
    velocity_rel : ndarray
        Centro de cada canal en km/s relativo a la velocidad sistemica.
    velocity_edges_rel : ndarray
        Bordes de los canales en km/s, tambien relativos a vsys.
    dv : ndarray
        Ancho positivo de cada canal en km/s.
    vlim : float
        Mitad del ancho espectral total. Se usa como limite simetrico del
        mapa de momento 1.
    """

    velocity = get_axis_centers_from_header(
        header=header,
        axis_number=spectral_axis_number,
        n_pix=n_channels,
        output_unit=u.km / u.s,
        rest_frequency=rest_frequency,
    )

    velocity_edges = centers_to_edges(velocity)

    # El punto medio del cubo corresponde a v_sys.
    v_sys = 0.5 * (velocity_edges[0] + velocity_edges[-1])

    velocity_rel = velocity - v_sys
    velocity_edges_rel = velocity_edges - v_sys

    dv = np.abs(np.diff(velocity_edges_rel))

    # Por construccion es exactamente la mitad del ancho total del cubo.
    vlim = 0.5 * np.abs(velocity_edges_rel[-1] - velocity_edges_rel[0])

    return velocity_rel, velocity_edges_rel, dv, vlim


def compute_moment_maps(
    cube_path,
    moment1_threshold=MOMENT1_THRESHOLD,
    rest_frequency=REST_FREQUENCY,
):
    """
    Calcula momento 0 y momento 1 a partir de un cubo FITS.

    Momento 0:
        M0 = sum I(v) dv

    Momento 1:
        M1 = sum I(v) (v-vsys) dv / sum I(v) dv

    Para M1 solo se incluyen voxeles con I >= MOMENT1_THRESHOLD.
    M0 se integra sobre todos los canales finitos sin aplicar ese threshold.
    """

    cube_path = Path(cube_path)

    with fits.open(cube_path) as hdul:
        cube = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header.copy()

    cube = np.squeeze(cube)

    if cube.ndim != 3:
        raise ValueError(
            f"El cubo {cube_path} no es 3D despues de squeeze. "
            f"Shape = {cube.shape}"
        )

    nchan, ny, nx = cube.shape

    velocity_rel, velocity_edges_rel, dv, vlim = get_cube_velocity_axis(
        header=header,
        n_channels=nchan,
        spectral_axis_number=3,
        rest_frequency=rest_frequency,
    )

    # --------------------------------------------------------
    # Momento 0: integra todos los canales finitos.
    # --------------------------------------------------------
    finite = np.isfinite(cube)
    cube_m0 = np.where(finite, cube, 0.0)

    moment0 = np.sum(
        cube_m0 * dv[:, np.newaxis, np.newaxis],
        axis=0,
    )

    # Si un pixel no tiene ningun dato finito, dejar NaN.
    moment0[~np.any(finite, axis=0)] = np.nan

    # --------------------------------------------------------
    # Momento 1: solo emision >= 5 sigma.
    # --------------------------------------------------------
    mask_m1 = finite & (cube >= moment1_threshold)

    weights = np.where(mask_m1, cube, 0.0) * dv[:, np.newaxis, np.newaxis]
    denominator = np.sum(weights, axis=0)
    numerator = np.sum(
        weights * velocity_rel[:, np.newaxis, np.newaxis],
        axis=0,
    )

    moment1 = np.full((ny, nx), np.nan, dtype=float)
    good = denominator > 0.0
    moment1[good] = numerator[good] / denominator[good]

    return {
        "moment0": moment0,
        "moment1": moment1,
        "header": header,
        "velocity_rel": velocity_rel,
        "velocity_edges_rel": velocity_edges_rel,
        "dv": dv,
        "vlim": vlim,
    }


def save_moment_fits(
    moment_data,
    input_header,
    output_path,
    moment_number,
    moment1_threshold=MOMENT1_THRESHOLD,
):
    """Guarda un mapa de momento como FITS 2D."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_bunit = input_header.get("BUNIT", "Jy/beam").strip()

    if moment_number == 0:
        if input_bunit:
            output_bunit = f"{input_bunit} km/s"
        else:
            output_bunit = "km/s"
    elif moment_number == 1:
        output_bunit = "km/s"
    else:
        raise ValueError("moment_number debe ser 0 o 1")

    out_header = make_spatial_2d_header(
        input_header=input_header,
        shape=moment_data.shape,
        bunit=output_bunit,
        moment_number=moment_number,
    )

    if moment_number == 1:
        out_header["THRESH"] = (
            float(moment1_threshold),
            "Intensity threshold used for moment 1",
        )
        out_header["THRUNIT"] = input_bunit if input_bunit else "Jy/beam"
        out_header["VELZERO"] = (0.0, "Velocity zero is cube spectral midpoint")

    fits.writeto(
        output_path,
        np.asarray(moment_data, dtype=np.float32),
        header=out_header,
        overwrite=True,
    )

    print(f"  FITS momento {moment_number} guardado: {output_path}")

    return output_path


def moment_to_png(
    moment_data,
    input_header,
    png_path,
    moment_number,
    vlim=None,
    percentile_clip=(1, 99),
):
    """Genera el PNG de un mapa de momento 0 o 1."""

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(moment_data, dtype=float)
    finite = np.isfinite(data)

    if not np.any(finite):
        print(f"  No se genera {png_path}: el mapa no tiene valores finitos.")
        return None

    ny, nx = data.shape

    x_edges = get_axis_edges_from_header(
        header=input_header,
        axis_number=1,
        n_pix=nx,
        output_unit=u.arcsec,
    )
    y_edges = get_axis_edges_from_header(
        header=input_header,
        axis_number=2,
        n_pix=ny,
        output_unit=u.arcsec,
    )

    x_edges = recenter_edges_to_middle(x_edges)
    y_edges = recenter_edges_to_middle(y_edges)

    extent = [
        x_edges[0],
        x_edges[-1],
        y_edges[0],
        y_edges[-1],
    ]

    plt.figure(figsize=(6, 5.5))

    if moment_number == 0:

        vmin, vmax = np.nanpercentile(data[finite], percentile_clip)

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = np.nanmin(data[finite])
            vmax = np.nanmax(data[finite])

        im = plt.imshow(
            data,
            origin="lower",
            cmap="inferno",
            aspect="equal",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )

        input_bunit = input_header.get("BUNIT", "Jy/beam").strip()
        cbar_label = rf"Momento 0 [{input_bunit} km s$^{{-1}}$]"
        title = "Momento 0"

    elif moment_number == 1:

        if vlim is None or not np.isfinite(vlim) or vlim <= 0:
            vlim = np.nanmax(np.abs(data[finite]))

        if not np.isfinite(vlim) or vlim <= 0:
            vlim = 1.0

        # RdBu_r: velocidades negativas -> azul; positivas -> rojo.
        norm = SymLogNorm(
            linthresh=0.5,
            linscale=1,
            vmin=-vlim,
            vmax=vlim,
            base=10,
        )
        im = plt.imshow(
            data,
            origin="lower",
            cmap="RdBu_r",
            aspect="equal",
            extent=extent,
            norm=norm,
        )

        cbar_label = r"Momento 1: $v-v_\mathrm{sys}$ [km s$^{-1}$]"
        title = "Momento 1"

    else:
        raise ValueError("moment_number debe ser 0 o 1")

    plt.colorbar(im, label=cbar_label)
    plt.xlabel(r"Offset X [arcsec]")
    plt.ylabel(r"Offset Y [arcsec]")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"  PNG momento {moment_number} guardado: {png_path}")

    return png_path


def generate_moments_for_model_folder(
    model_folder,
    group_dir,
    moment1_threshold=MOMENT1_THRESHOLD,
):
    """
    Genera y guarda M0/M1 para una sola carpeta de modelo.

    Estructura de salida:

        <grupo>/moments/moment0/fits/
        <grupo>/moments/moment0/png/
        <grupo>/moments/moment1/fits/
        <grupo>/moments/moment1/png/
    """

    model_folder = Path(model_folder)
    group_dir = Path(group_dir)

    cube_path = find_cube_file(model_folder)

    if cube_path is None:
        print(
            f"  No encontre cubo en {model_folder}. "
            f"Busque: {', '.join(CUBE_CANDIDATES)}"
        )
        return []

    print(f"  Generando momentos desde: {cube_path}")

    result = compute_moment_maps(
        cube_path=cube_path,
        moment1_threshold=moment1_threshold,
        rest_frequency=REST_FREQUENCY,
    )

    generated = []

    for moment_number in (0, 1):

        moment_key = f"moment{moment_number}"
        moment_dir = group_dir / "moments" / moment_key
        fits_dir = moment_dir / "fits"
        png_dir = moment_dir / "png"

        fits_path = fits_dir / moment_name(
            model_folder.name,
            moment_number,
            extension="fits",
        )
        png_path = png_dir / moment_name(
            model_folder.name,
            moment_number,
            extension="png",
        )

        save_moment_fits(
            moment_data=result[moment_key],
            input_header=result["header"],
            output_path=fits_path,
            moment_number=moment_number,
            moment1_threshold=moment1_threshold,
        )

        moment_to_png(
            moment_data=result[moment_key],
            input_header=result["header"],
            png_path=png_path,
            moment_number=moment_number,
            vlim=result["vlim"] if moment_number == 1 else None,
        )

        generated.extend([fits_path, png_path])

    return generated


def generate_all_moment_maps(
    base_dir=BASE_DIR,
    output_dir=OUTPUT_DIR,
    moment1_threshold=MOMENT1_THRESHOLD,
):
    """Genera mapas de momentos para todos los modelos definidos en GROUPS."""

    base_dir = Path(base_dir)
    output_dir = Path(output_dir)

    generated_files = []

    print("\nGenerando mapas de momentos 0 y 1...")
    print(f"Threshold momento 1: {moment1_threshold:.3e} Jy/beam")

    for group_name, pattern in GROUPS.items():

        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        folders = sorted(base_dir.glob(pattern))

        print(f"\nGrupo de momentos: {group_name}")
        print(f"Carpetas encontradas: {len(folders)}")

        for folder in folders:

            if not folder.is_dir():
                continue

            try:
                generated = generate_moments_for_model_folder(
                    model_folder=folder,
                    group_dir=group_dir,
                    moment1_threshold=moment1_threshold,
                )
                generated_files.extend(generated)

            except Exception as e:
                print(f"  ERROR generando momentos para {folder}: {e}")

    return generated_files


# ============================================================
# Ejecucion principal
# ============================================================

if __name__ == "__main__":

    copied = copy_pv_files()
    pngs = convert_all_copied_fits_to_png()
    moments = generate_all_moment_maps()

    n_moment_fits = sum(Path(path).suffix.lower() == ".fits" for path in moments)
    n_moment_png = sum(Path(path).suffix.lower() == ".png" for path in moments)

    print("\nResumen:")
    print(f"FITS PV copiados: {len(copied)}")
    print(f"PNG PV generados: {len(pngs)}")
    print(f"FITS de momentos generados: {n_moment_fits}")
    print(f"PNG de momentos generados: {n_moment_png}")
    print(f"Salida en: {OUTPUT_DIR}")
