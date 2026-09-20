#!/usr/bin/env python3

"""
Create a YSO-model configuration JSON from:

    1. An observational FITS cube
    2. A PV FITS extracted from that cube

The generated JSON follows the structure used by the current
MCMC / yso_models / make_line pipeline.

Main conventions
----------------

1. The synthetic RADMC image has 4 more spatial pixels than
   the observational PV:

       npix_radmc = npix_pv + 4

2. The physical model grid has the same number of cells as
   the RADMC image.

3. The physical cell size is equal to the angular pixel size
   of the observational cube converted to AU:

       cell_size_au = pixel_scale_arcsec * distance_pc

4. Therefore:

       half_size_au = 0.5 * npix_radmc * cell_size_au

5. The synthetic number of spectral channels is exactly equal
   to the observational PV:

       nchan = nchan_pv

6. The velocity channel width is taken from the PV header.

7. The synthetic PV spacing is taken from the PV header.

8. A tiny margin is added to the PV length to prevent
   pvextractor from dropping one spatial pixel:

       pv_length =
           npix_pv * spacing
           + 0.01 * spacing
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import astropy.units as u

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.constants import c


# ============================================================
# USER INPUT
# ============================================================

def ask_float(message, default=None):
    """
    Ask interactively for a floating-point value.
    """

    if default is None:

        while True:

            text = input(
                f"{message}: "
            ).strip()

            try:
                return float(text)

            except ValueError:
                print(
                    "Please enter a valid number."
                )

    else:

        text = input(
            f"{message} [{default}]: "
        ).strip()

        if text == "":
            return float(default)

        return float(text)


def ask_int(message, default=None):
    """
    Ask interactively for an integer.
    """

    if default is None:

        while True:

            text = input(
                f"{message}: "
            ).strip()

            try:
                return int(text)

            except ValueError:
                print(
                    "Please enter a valid integer."
                )

    else:

        text = input(
            f"{message} [{default}]: "
        ).strip()

        if text == "":
            return int(default)

        return int(text)


def ask_string(message, default=None):
    """
    Ask interactively for a string.
    """

    if default is None:

        while True:

            text = input(
                f"{message}: "
            ).strip()

            if text:
                return text

    else:

        text = input(
            f"{message} [{default}]: "
        ).strip()

        if text == "":
            return str(default)

        return text


# ============================================================
# FITS
# ============================================================

def get_header(filename):
    """
    Read primary FITS header.
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"FITS file not found: {filename}"
        )

    with fits.open(filename) as hdul:

        header = hdul[0].header.copy()

    return header


# ============================================================
# AXIS IDENTIFICATION
# ============================================================

def find_spectral_axis(header):
    """
    Find the FITS spectral axis number.

    Recognized axes:
        FREQ
        VRAD
        VELO
        VOPT
    """

    spectral_types = (
        "FREQ",
        "VRAD",
        "VELO",
        "VOPT"
    )

    naxis = int(
        header["NAXIS"]
    )

    for axis in range(
        1,
        naxis + 1
    ):

        ctype = str(
            header.get(
                f"CTYPE{axis}",
                ""
            )
        ).upper()

        if any(
            name in ctype
            for name in spectral_types
        ):
            return axis

    raise ValueError(
        "Could not identify the spectral axis."
    )


def find_pv_spatial_axis(header):
    """
    Find the spatial OFFSET axis in a PV diagram.
    """

    naxis = int(
        header["NAXIS"]
    )

    # Prefer explicit OFFSET axis
    for axis in range(
        1,
        naxis + 1
    ):

        ctype = str(
            header.get(
                f"CTYPE{axis}",
                ""
            )
        ).upper()

        if "OFFSET" in ctype:
            return axis

    # Otherwise use the non-spectral axis
    spectral_axis = find_spectral_axis(
        header
    )

    for axis in range(
        1,
        naxis + 1
    ):

        if axis != spectral_axis:
            return axis

    raise ValueError(
        "Could not identify the PV spatial axis."
    )


# ============================================================
# FREQUENCY
# ============================================================

def get_rest_frequency(header):
    """
    Try common FITS rest-frequency keywords.
    """

    possible_keys = (
        "RESTFRQ",
        "RESTFREQ"
    )

    for key in possible_keys:

        if key in header:

            value = float(
                header[key]
            )

            if value > 0:
                return value

    return None


# ============================================================
# COORDINATES
# ============================================================

def get_source_coordinates(header):
    """
    Determine source RA/Dec.

    Priority:
        1. OBSRA / OBSDEC
        2. Celestial WCS CRVAL
        3. Ask user
    """

    # --------------------------------------------------------
    # OBSRA / OBSDEC
    # --------------------------------------------------------

    if (
        "OBSRA" in header
        and "OBSDEC" in header
    ):

        coord = SkyCoord(
            float(header["OBSRA"])
            * u.deg,

            float(header["OBSDEC"])
            * u.deg,

            frame="icrs"
        )

    else:

        # ----------------------------------------------------
        # Try celestial WCS
        # ----------------------------------------------------

        try:

            celestial = WCS(
                header
            ).celestial

            world = (
                celestial.wcs.crval
            )

            coord = SkyCoord(
                world[0] * u.deg,
                world[1] * u.deg,
                frame="icrs"
            )

        except Exception:

            print(
                "\nCould not determine RA/Dec "
                "from the cube header."
            )

            ra_string = ask_string(
                "RA "
                "(example: 16h29m46.13s)"
            )

            dec_string = ask_string(
                "Dec "
                "(example: -48d15m49.95s)"
            )

            return (
                ra_string,
                dec_string
            )

    # --------------------------------------------------------
    # Convert to strings compatible with SkyCoord
    # --------------------------------------------------------

    ra_string = (
        coord.ra.to_string(
            unit=u.hourangle,
            sep="hms",
            precision=5,
            pad=True
        )
    )

    dec_string = (
        coord.dec.to_string(
            unit=u.deg,
            sep="dms",
            precision=4,
            pad=True,
            alwayssign=True
        )
    )

    return (
        ra_string,
        dec_string
    )


# ============================================================
# CUBE PIXEL SCALE
# ============================================================

def get_cube_pixel_scale(header):
    """
    Determine spatial pixel scale of observational cube
    in arcsec/pixel.

    If pixels are not square, ask the user which scale
    should be used for the model.
    """

    try:

        wcs = WCS(
            header
        )

        celestial = wcs.celestial

        scales_deg = (
            proj_plane_pixel_scales(
                celestial
            )
        )

        sx = (
            abs(scales_deg[0])
            * u.deg
        ).to_value(
            u.arcsec
        )

        sy = (
            abs(scales_deg[1])
            * u.deg
        ).to_value(
            u.arcsec
        )

    except Exception:

        print(
            "\nCould not determine the "
            "spatial pixel size from "
            "the cube WCS."
        )

        return ask_float(
            "Cube spatial pixel size "
            "[arcsec/pixel]"
        )

    print(
        "\nCube spatial pixels:"
    )

    print(
        f"    X = {sx:.10f} arcsec"
    )

    print(
        f"    Y = {sy:.10f} arcsec"
    )

    # --------------------------------------------------------
    # Square pixels
    # --------------------------------------------------------

    if np.isclose(
        sx,
        sy,
        rtol=1e-4
    ):

        scale = (
            sx + sy
        ) / 2.0

        return scale

    # --------------------------------------------------------
    # Non-square pixels
    # --------------------------------------------------------

    print(
        "\nWARNING:"
        "\nThe cube pixels are not square."
    )

    scale = ask_float(
        "Spatial pixel size to use "
        "for the model [arcsec]",
        default=min(
            sx,
            sy
        )
    )

    return scale


# ============================================================
# PV SPATIAL SAMPLING
# ============================================================

def get_pv_spatial_sampling(header):
    """
    Return:

        spacing_arcsec
        npix_pv
    """

    axis = find_pv_spatial_axis(
        header
    )

    cdelt = abs(
        float(
            header[
                f"CDELT{axis}"
            ]
        )
    )

    unit_string = str(
        header.get(
            f"CUNIT{axis}",
            "deg"
        )
    )

    try:

        unit = u.Unit(
            unit_string
        )

        spacing = (
            cdelt * unit
        ).to_value(
            u.arcsec
        )

    except Exception:

        print(
            "\nCould not interpret "
            f"CUNIT{axis} = {unit_string}"
        )

        spacing = ask_float(
            "PV spatial spacing "
            "[arcsec/pixel]"
        )

    npix = int(
        header[
            f"NAXIS{axis}"
        ]
    )

    return (
        spacing,
        npix
    )


# ============================================================
# PV SPECTRAL SAMPLING
# ============================================================

def get_pv_spectral_sampling(
    header,
    restfreq_hz
):
    """
    Determine:

        dv_kms
        nchan

    Handles velocity axes and frequency axes.
    """

    axis = find_spectral_axis(
        header
    )

    ctype = str(
        header.get(
            f"CTYPE{axis}",
            ""
        )
    ).upper()

    cdelt = abs(
        float(
            header[
                f"CDELT{axis}"
            ]
        )
    )

    unit_string = str(
        header.get(
            f"CUNIT{axis}",
            ""
        )
    )

    nchan = int(
        header[
            f"NAXIS{axis}"
        ]
    )

    # --------------------------------------------------------
    # Velocity axis
    # --------------------------------------------------------

    try:

        spectral_unit = u.Unit(
            unit_string
        )

    except Exception:

        spectral_unit = None

    if (
        spectral_unit is not None
        and
        spectral_unit.is_equivalent(
            u.m / u.s
        )
    ):

        dv_kms = (
            cdelt
            * spectral_unit
        ).to_value(
            u.km / u.s
        )

        return (
            dv_kms,
            nchan
        )

    # --------------------------------------------------------
    # Frequency axis
    # --------------------------------------------------------

    if (
        spectral_unit is not None
        and
        spectral_unit.is_equivalent(
            u.Hz
        )
    ):

        if restfreq_hz is None:

            print(
                "\nPV uses a frequency axis, "
                "but no RESTFRQ was found."
            )

            restfreq_hz = ask_float(
                "Rest frequency [Hz]"
            )

        delta_nu_hz = (
            cdelt
            * spectral_unit
        ).to_value(
            u.Hz
        )

        c_kms = c.to_value(
            u.km / u.s
        )

        # Radio approximation:
        #
        # dv / c = dnu / nu0

        dv_kms = (
            c_kms
            * delta_nu_hz
            / restfreq_hz
        )

        return (
            dv_kms,
            nchan
        )

    raise ValueError(
        "\nCould not determine velocity "
        "channel width from PV.\n"
        f"CTYPE{axis} = {ctype}\n"
        f"CUNIT{axis} = {unit_string}"
    )


# ============================================================
# BEAM
# ============================================================

def get_beam(header):
    """
    Get synthesized beam from observational cube.

    BMAJ/BMIN are normally stored in degrees.
    """

    if (
        "BMAJ" in header
        and
        "BMIN" in header
    ):

        bmaj = (
            float(
                header["BMAJ"]
            )
            * u.deg
        ).to_value(
            u.arcsec
        )

        bmin = (
            float(
                header["BMIN"]
            )
            * u.deg
        ).to_value(
            u.arcsec
        )

        if "BPA" in header:

            bpa = float(
                header["BPA"]
            )

        else:

            print(
                "\nBMAJ and BMIN were found, "
                "but BPA is missing."
            )

            bpa = ask_float(
                "Beam PA [deg]"
            )

        return (
            bmaj,
            bmin,
            bpa
        )

    # --------------------------------------------------------
    # Beam missing
    # --------------------------------------------------------

    print(
        "\nBeam information was not "
        "found in the primary cube header."
    )

    bmaj = ask_float(
        "Beam major axis [arcsec]"
    )

    bmin = ask_float(
        "Beam minor axis [arcsec]"
    )

    bpa = ask_float(
        "Beam PA [deg]"
    )

    return (
        bmaj,
        bmin,
        bpa
    )


# ============================================================
# DEFAULT CONFIG
# ============================================================

def default_config():
    """
    Default non-observational configuration.

    These values can be replaced by loading an existing
    JSON with --template.
    """

    return {

        "model_options": {

            "disc": True,

            "envelope": True,

            "gas_to_dust_ratio":
                100.0,

            "microturbulence_ms":
                100.0,

            "rho_min_env_m3":
                1e10
        },

        "radmc3d": {

            "nphot":
                100000000,

            "threads":
                1,

            "incl_dust":
                1,

            "incl_freefree":
                0,

            "tgas_eq_tdust":
                0,

            "modified_random_walk":
                0,

            "wavelength_intervals_micron": [
                0.1,
                500.0,
                10000.0
            ],

            "wavelength_divisions": [
                20,
                20
            ]
        },

        "mcmc": {

            "nthreads":
                1,

            "omp_threads":
                2,

            "nwalkers":
                32,

            "nburn":
                100,

            "nsteps":
                1000,

            "frac_stddev":
                0.01,

            "parameter_file":
                "free_params_ulrich.csv"
        },

        "output": {

            "tag":
                "mcmc",

            "test_number":
                1
        }
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Generate a modelling JSON "
            "from an observational cube "
            "and PV diagram."
        )
    )

    parser.add_argument(
        "cube",
        help=(
            "Observational FITS cube"
        )
    )

    parser.add_argument(
        "pv",
        help=(
            "PV FITS generated from "
            "the observational cube"
        )
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output JSON filename"
        )
    )

    parser.add_argument(
        "--template",
        default=None,
        help=(
            "Existing configuration JSON. "
            "Non-observational options such "
            "as MCMC and RADMC settings will "
            "be reused."
        )
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=4,
        help=(
            "Extra spatial pixels in the "
            "RADMC image relative to the PV. "
            "Default = 4."
        )
    )

    args = parser.parse_args()

    cube_file = Path(
        args.cube
    )

    pv_file = Path(
        args.pv
    )

    # --------------------------------------------------------
    # Read FITS headers
    # --------------------------------------------------------

    cube_header = get_header(
        cube_file
    )

    pv_header = get_header(
        pv_file
    )

    # ========================================================
    # BASE CONFIG
    # ========================================================

    if args.template is not None:

        template_file = Path(
            args.template
        )

        with open(
            template_file,
            "r"
        ) as f:

            config = json.load(
                f
            )

        config = copy.deepcopy(
            config
        )

        print(
            "\nUsing template:"
        )

        print(
            f"    {template_file}"
        )

    else:

        config = default_config()

    # ========================================================
    # SOURCE NAME
    # ========================================================

    object_name = str(
        cube_header.get(
            "OBJECT",
            cube_file.stem
        )
    ).strip()

    print(
        "\n=================================="
    )

    print(
        f"Object: {object_name}"
    )

    print(
        "=================================="
    )

    # ========================================================
    # COORDINATES
    # ========================================================

    ra, dec = (
        get_source_coordinates(
            cube_header
        )
    )

    print(
        "\nCoordinates:"
    )

    print(
        f"    RA  = {ra}"
    )

    print(
        f"    Dec = {dec}"
    )

    # ========================================================
    # PARAMETERS NOT RELIABLY IN FITS
    # ========================================================

    print(
        "\n----------------------------------"
    )

    print(
        "Required user parameters"
    )

    print(
        "----------------------------------"
    )

    distance_pc = ask_float(
        "Source distance [pc]"
    )

    v_sys = ask_float(
        "Systemic velocity [km/s]"
    )

    # ========================================================
    # FREQUENCY
    # ========================================================

    cube_restfreq = (
        get_rest_frequency(
            cube_header
        )
    )

    pv_restfreq = (
        get_rest_frequency(
            pv_header
        )
    )

    print(
        "\nFrequency information:"
    )

    if cube_restfreq is not None:

        print(
            "    Cube RESTFRQ = "
            f"{cube_restfreq:.6f} Hz"
        )

    else:

        print(
            "    Cube RESTFRQ = not found"
        )

    if pv_restfreq is not None:

        print(
            "    PV RESTFRQ   = "
            f"{pv_restfreq:.6f} Hz"
        )

    else:

        print(
            "    PV RESTFRQ   = not found"
        )

    # --------------------------------------------------------
    # Decide line frequency
    # --------------------------------------------------------

    if pv_restfreq is not None:

        linefreq_hz = (
            pv_restfreq
        )

    elif cube_restfreq is not None:

        linefreq_hz = (
            cube_restfreq
        )

    else:

        print(
            "\nNo RESTFRQ was found "
            "in either FITS header."
        )

        linefreq_hz = ask_float(
            "Line rest frequency [Hz]"
        )

    # --------------------------------------------------------
    # restfreq used by make_line
    # --------------------------------------------------------

    if cube_restfreq is not None:

        restfreq_hz = (
            cube_restfreq
        )

    else:

        restfreq_hz = (
            linefreq_hz
        )

    # --------------------------------------------------------
    # If the two headers disagree, inform user
    # --------------------------------------------------------

    if (
        cube_restfreq is not None
        and
        pv_restfreq is not None
        and
        not np.isclose(
            cube_restfreq,
            pv_restfreq,
            rtol=0.0,
            atol=1.0e3
        )
    ):

        print(
            "\nWARNING:"
        )

        print(
            "Cube and PV contain different "
            "RESTFRQ values."
        )

        print(
            f"    cube = "
            f"{cube_restfreq:.6f} Hz"
        )

        print(
            f"    PV   = "
            f"{pv_restfreq:.6f} Hz"
        )

        restfreq_hz = ask_float(
            "restfreq_hz",
            default=cube_restfreq
        )

        linefreq_hz = ask_float(
            "linefreq_hz",
            default=pv_restfreq
        )

    # ========================================================
    # MOLECULE
    # ========================================================

    default_molecule = (
        config
        .get(
            "line",
            {}
        )
        .get(
            "molecule",
            "ch3oh"
        )
    )

    default_transition = (
        config
        .get(
            "line",
            {}
        )
        .get(
            "transition",
            1
        )
    )

    molecule = ask_string(
        "Molecule",
        default=default_molecule
    )

    transition = ask_int(
        "RADMC transition number",
        default=default_transition
    )

    # ========================================================
    # CUBE SPATIAL PIXEL
    # ========================================================

    cube_pixel_arcsec = (
        get_cube_pixel_scale(
            cube_header
        )
    )

    # ========================================================
    # PV SPATIAL SAMPLING
    # ========================================================

    (
        pv_spacing_arcsec,
        pv_npix
    ) = get_pv_spatial_sampling(
        pv_header
    )

    print(
        "\nSpatial sampling:"
    )

    print(
        "    Cube pixel = "
        f"{cube_pixel_arcsec:.10f}\""
    )

    print(
        "    PV spacing = "
        f"{pv_spacing_arcsec:.10f}\""
    )

    print(
        "    PV pixels  = "
        f"{pv_npix}"
    )

    if not np.isclose(
        cube_pixel_arcsec,
        pv_spacing_arcsec,
        rtol=1e-4
    ):

        print(
            "\nWARNING:"
        )

        print(
            "The spatial pixel size of "
            "the cube and the PV spacing "
            "are different."
        )

    # ========================================================
    # SYNTHETIC IMAGE SIZE
    # ========================================================

    image_npix = (
        pv_npix
        + args.padding
    )

    # ========================================================
    # PHYSICAL GRID
    # ========================================================

    # 1 arcsec at 1 pc = 1 AU

    cell_size_au = (
        cube_pixel_arcsec
        * distance_pc
    )

    # Grid width:
    #
    #     Ncells * cell_size
    #
    # Grid half-size:
    #
    #     width / 2

    grid_size_au = (
        image_npix
        * cell_size_au
    )

    half_size_au = (
        grid_size_au
        / 2.0
    )

    # ========================================================
    # SPECTRAL SAMPLING
    # ========================================================

    (
        dv_kms,
        pv_nchan
    ) = get_pv_spectral_sampling(
        pv_header,
        linefreq_hz
    )

    # IMPORTANT:
    #
    # Number of synthetic channels is
    # exactly equal to N_PV.

    image_nchan = (
        pv_nchan
    )

    # ========================================================
    # BEAM
    # ========================================================

    (
        bmaj,
        bmin,
        bpa
    ) = get_beam(
        cube_header
    )

    # ========================================================
    # PV GEOMETRY
    # ========================================================

    print(
        "\n----------------------------------"
    )

    print(
        "PV geometry"
    )

    print(
        "----------------------------------"
    )

    print(
        "The PV FITS header normally "
        "does not preserve the PA and "
        "slit width."
    )

    pv_pa = ask_float(
        "PV position angle [deg]"
    )

    pv_width = ask_float(
        "PV slit width [arcsec]"
    )

    # --------------------------------------------------------
    # Tiny margin:
    #
    # Example:
    #
    #   99 * 0.01 = 0.99
    #
    # becomes
    #
    #   0.9901
    #
    # This prevents losing one pixel due
    # to floating-point / extraction limits.
    # --------------------------------------------------------

    pv_length = (
        pv_npix
        * pv_spacing_arcsec
        +
        0.01
        * pv_spacing_arcsec
    )

    # ========================================================
    # FILE PATHS
    # ========================================================

    default_noise_file = (
        f"inputs/"
        f"{object_name}_noise.dat"
    )

    noise_file = ask_string(
        "Noise file",
        default=default_noise_file
    )

    parameter_default = (
        config
        .get(
            "mcmc",
            {}
        )
        .get(
            "parameter_file",
            "free_params_ulrich.csv"
        )
    )

    parameter_file = ask_string(
        "MCMC parameter CSV",
        default=parameter_default
    )

    # ========================================================
    # BUILD CONFIG
    # ========================================================

    config["source"] = {

        "name":
            object_name,

        "distance_pc":
            distance_pc,

        "ra":
            ra,

        "dec":
            dec,

        "v_sys_kms":
            v_sys
    }

    config["line"] = {

        "molecule":
            molecule,

        "transition":
            transition,

        "restfreq_hz":
            restfreq_hz,

        "linefreq_hz":
            linefreq_hz
    }

    config["observation"] = {

        "pv_file":
            str(pv_file),

        "noise_file":
            noise_file
    }

    config["physical_grid"] = {

        "half_size_au": [
            half_size_au,
            half_size_au,
            half_size_au
        ],

        "npoints": [
            image_npix,
            image_npix,
            image_npix
        ],

        "include_zero":
            True
    }

    config["synthetic_image"] = {

        "npix":
            image_npix,

        "nchan":
            image_nchan,

        "dv_kms":
            dv_kms,

        "pixel_scale_arcsec":
            cube_pixel_arcsec
    }

    config["beam"] = {

        "major_arcsec":
            bmaj,

        "minor_arcsec":
            bmin,

        "pa_deg":
            bpa
    }

    config["pv"] = {

        "pa_deg":
            pv_pa,

        "length_arcsec":
            pv_length,

        "width_arcsec":
            pv_width,

        "spacing_arcsec":
            pv_spacing_arcsec
    }

    config["mcmc"][
        "parameter_file"
    ] = parameter_file

    config["output"]["tag"] = (
        f"mcmc_{object_name}"
    )

    # ========================================================
    # OUTPUT FILE
    # ========================================================

    if args.output is None:

        safe_name = (
            object_name
            .replace(
                " ",
                "_"
            )
            .replace(
                "/",
                "_"
            )
        )

        output_file = Path(
            f"configs/{safe_name}.json"
        )

    else:

        output_file = Path(
            args.output
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        "=================================="
    )

    print(
        "CONFIGURATION CREATED"
    )

    print(
        "=================================="
    )

    print(
        f"\nJSON:"
        f"\n    {output_file}"
    )

    print(
        "\nSource:"
    )

    print(
        f"    Name      = "
        f"{object_name}"
    )

    print(
        f"    Distance  = "
        f"{distance_pc:.3f} pc"
    )

    print(
        f"    RA        = "
        f"{ra}"
    )

    print(
        f"    Dec       = "
        f"{dec}"
    )

    print(
        f"    Vsys      = "
        f"{v_sys:.6f} km/s"
    )

    print(
        "\nSpatial:"
    )

    print(
        f"    Cube pixel        = "
        f"{cube_pixel_arcsec:.8f}\""
    )

    print(
        f"    PV spacing        = "
        f"{pv_spacing_arcsec:.8f}\""
    )

    print(
        f"    Observed PV npix  = "
        f"{pv_npix}"
    )

    print(
        f"    RADMC npix        = "
        f"{image_npix}"
    )

    print(
        f"    Grid              = "
        f"{image_npix} x "
        f"{image_npix} x "
        f"{image_npix}"
    )

    print(
        f"    Cell size         = "
        f"{cell_size_au:.6f} AU"
    )

    print(
        f"    Grid full size    = "
        f"{grid_size_au:.6f} AU"
    )

    print(
        f"    Grid half-size    = "
        f"{half_size_au:.6f} AU"
    )

    print(
        "\nSpectral:"
    )

    print(
        f"    PV channels       = "
        f"{pv_nchan}"
    )

    print(
        f"    Synthetic nchan   = "
        f"{image_nchan}"
    )

    print(
        f"    dv                = "
        f"{dv_kms:.9f} km/s"
    )

    print(
        f"    restfreq          = "
        f"{restfreq_hz:.6f} Hz"
    )

    print(
        f"    linefreq          = "
        f"{linefreq_hz:.6f} Hz"
    )

    print(
        "\nPV:"
    )

    print(
        f"    PA                = "
        f"{pv_pa:.6f} deg"
    )

    print(
        f"    Width             = "
        f"{pv_width:.6f}\""
    )

    print(
        f"    Length            = "
        f"{pv_length:.10f}\""
    )

    print(
        f"    Spacing           = "
        f"{pv_spacing_arcsec:.10f}\""
    )

    print(
        "\nBeam:"
    )

    print(
        f"    BMAJ              = "
        f"{bmaj:.6f}\""
    )

    print(
        f"    BMIN              = "
        f"{bmin:.6f}\""
    )

    print(
        f"    BPA               = "
        f"{bpa:.6f} deg"
    )

    print(
        "\nConsistency:"
    )

    print(
        f"    RADMC padding     = "
        f"{args.padding} pixels"
    )

    print(
        "    nchan synthetic   = "
        "nchan observational"
    )

    print(
        "    grid cell size    = "
        "cube spatial pixel"
    )

    print(
        "=================================="
    )


if __name__ == "__main__":
    main()