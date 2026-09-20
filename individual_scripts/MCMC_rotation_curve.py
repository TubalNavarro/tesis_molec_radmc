#!/usr/bin/env python3
"""
MCMC fit of a PV rotation-curve envelope with

    central point mass + spherical envelope + axisymmetric flared/Pringle disk

The observed rotation curve is extracted directly from a 2-D PV FITS:
  * left half: first >= N sigma pixel from top -> bottom
  * right half: first >= N sigma pixel from bottom -> top

The MCMC fits the curve only. No sf3dmodels, RADMC-3D, synthetic cubes, or
per-model folders are generated.

Free/fixed parameters are read from a CSV with columns:
    Parameter, Mean, Min, Max, Fit

Supported model parameters:
    MStar      [Msun]  central point mass
    Mdisk      [Msun]  total disk mass between Rstar and Rdisk
    Menv       [Msun]  total spherical-envelope mass inside Rout
    diskPower          exponent q in rho_mid(R) proportional to R^q
    Rdisk       [AU]   outer radius of the disk
    envPower           exponent p in rho_env(r) proportional to r^-p
    incl        [deg]  inclination, 0=face-on, 90=edge-on

The expensive elliptic-integral kernel of the disk is computed only once.
During the MCMC, changing diskPower and Rdisk only changes the radial weights
of that precomputed kernel, while Mdisk provides the total-mass normalization.
This keeps the likelihood fast even when the disk density slope and size are
free parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from multiprocessing import Pool

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.special import ellipk, ellipe

# User's plotting helpers. If unavailable, corner/walker plotting is skipped
# gracefully while the best-fit curve is still generated.
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from plot_helpers import plot_walkers, plot_corner
try:
    from plot_helpers import plot_walkers, plot_corner
    HAVE_PLOT_HELPERS = True
except ImportError:
    HAVE_PLOT_HELPERS = False


# =============================================================================
# Constants
# =============================================================================
G = 6.67430e-11
M_SUN = 1.98847e30
M_H = 1.6735575e-27
M_H2 = 2.0 * M_H
AU = 1.495978707e11
R_SUN = 6.957e8
R_SUN_TO_AU = R_SUN / AU
C_KMS = 299792.458
PC_TO_AU = 206264.80624709636
V_1AU_1MSUN = np.sqrt(G * M_SUN / AU) / 1000.0


# =============================================================================
# FITS axes and -envelope extraction
# =============================================================================
def linear_axis_from_header(header, fits_axis: int, n: int) -> np.ndarray:
    crval = float(header.get(f"CRVAL{fits_axis}", 0.0))
    crpix = float(header.get(f"CRPIX{fits_axis}", 1.0))
    cdelt = header.get(f"CDELT{fits_axis}")
    if cdelt is None:
        cdelt = header.get(f"CD{fits_axis}_{fits_axis}")
    if cdelt is None:
        raise KeyError(
            f"No encuentro CDELT{fits_axis} ni CD{fits_axis}_{fits_axis}."
        )
    pix = np.arange(n, dtype=float) + 1.0
    return crval + (pix - crpix) * float(cdelt)


def spatial_axis_arcsec(header, ncols: int) -> np.ndarray:
    x = linear_axis_from_header(header, 1, ncols)
    unit = str(header.get("CUNIT1", "")).strip().lower().replace(" ", "")
    if unit in {"deg", "degree", "degrees"}:
        x *= 3600.0
    elif unit in {"arcsec", "arcsecond", "arcseconds", "asec", '"'}:
        pass
    elif unit in {"arcmin", "arcminute", "arcminutes"}:
        x *= 60.0
    elif unit in {"rad", "radian", "radians"}:
        x = np.rad2deg(x) * 3600.0
    elif unit == "":
        print("ADVERTENCIA: CUNIT1 vacío; asumiré arcsec.")
    else:
        raise ValueError(f"CUNIT1='{unit}' no soportado.")
    return x


def velocity_axis_kms(header, nrows: int, rest_freq_ghz=None) -> np.ndarray:
    """Return FITS axis 2 in km/s.

    For CTYPE2=FREQ use radio convention:
        v = c (nu0 - nu) / nu0
    """
    y = linear_axis_from_header(header, 2, nrows)
    unit = str(header.get("CUNIT2", "")).strip().lower().replace(" ", "")
    ctype = str(header.get("CTYPE2", "")).upper()

    if "VELO" in ctype or "VRAD" in ctype:
        if unit in {"km/s", "kms-1", "km.s-1"}:
            return y
        return y / 1000.0

    if "FREQ" in ctype:
        scale = {
            "hz": 1.0,
            "khz": 1e3,
            "mhz": 1e6,
            "ghz": 1e9,
            "": 1.0,
        }.get(unit)
        if scale is None:
            raise ValueError(f"CUNIT2='{unit}' no soportado para frecuencia.")
        nu_hz = y * scale
        if rest_freq_ghz is not None:
            nu0_hz = float(rest_freq_ghz) * 1e9
        else:
            nu0_hz = header.get("RESTFRQ", header.get("RESTFREQ"))
            if nu0_hz is None:
                raise ValueError("No encuentro RESTFRQ/RESTFREQ.")
            nu0_hz = float(nu0_hz)
        return C_KMS * (nu0_hz - nu_hz) / nu0_hz

    raise ValueError(
        f"No puedo interpretar CTYPE2='{ctype}', CUNIT2='{unit}'."
    )


def spectral_center_velocity(velocity_kms: np.ndarray) -> float:
    row = (len(velocity_kms) - 1) / 2.0
    return float(np.interp(row, np.arange(len(velocity_kms)), velocity_kms))


def parse_ranges(items) -> list[tuple[int, int]]:
    if not items:
        return []
    out = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            lo, hi = map(int, item)
        else:
            lo, hi = map(int, str(item).split(":", 1))
        if lo > hi:
            lo, hi = hi, lo
        out.append((lo, hi))
    return out


def columns_in_ranges(cols, ranges):
    mask = np.zeros(np.shape(cols), dtype=bool)
    for lo, hi in ranges:
        mask |= (cols >= lo) & (cols <= hi)
    return mask


def extract_envelope(data, threshold, center_col, velocity_rel_kms):
    """Extract exactly the PV edge discussed in the previous script."""
    nvel, nspace = data.shape
    finite_vel = np.isfinite(velocity_rel_kms)

    # Visual top = largest physical velocity; bottom = smallest.
    top_to_bottom = np.argsort(
        np.where(finite_vel, velocity_rel_kms, -np.inf)
    )[::-1]
    bottom_to_top = np.argsort(
        np.where(finite_vel, velocity_rel_kms, np.inf)
    )

    rows, cols, sides = [], [], []

    # Left: top -> bottom
    for col in range(center_col):
        for row in top_to_bottom:
            val = data[row, col]
            if finite_vel[row] and np.isfinite(val) and val >= threshold:
                rows.append(int(row))
                cols.append(int(col))
                sides.append(-1)
                break

    # Right: bottom -> top
    for col in range(center_col + 1, nspace):
        for row in bottom_to_top:
            val = data[row, col]
            if finite_vel[row] and np.isfinite(val) and val >= threshold:
                rows.append(int(row))
                cols.append(int(col))
                sides.append(+1)
                break

    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(sides, dtype=int),
    )


# =============================================================================
# Spherical envelope
# =============================================================================
def envelope_enclosed_mass_msun(r_au, total_mass_msun, power, rout_au):
    """M_env(<r) for rho proportional to r^-power.

    Menv parameter is defined as the TOTAL envelope mass inside Rout.
    For p < 3:
        M(<r) = Menv * (r/Rout)^(3-p), r <= Rout.
    """
    if total_mass_msun < 0:
        raise ValueError("Menv must be >= 0")
    if rout_au <= 0:
        raise ValueError("Envelope Rout must be > 0")
    if power >= 3:
        raise ValueError("Envelope power must be < 3")

    r = np.asarray(r_au, dtype=float)
    reff = np.minimum(r, rout_au)
    return total_mass_msun * (reff / rout_au) ** (3.0 - power)


def envelope_n0_cm3_from_total_mass(
    total_mass_msun, power, rout_au, r0_au=3600.0
):
    """Density normalization n(r0) implied by total envelope mass."""
    mass_kg = total_mass_msun * M_SUN
    r0_m = r0_au * AU
    rout_m = rout_au * AU
    rho0 = (
        mass_kg
        * (3.0 - power)
        / (4.0 * np.pi * r0_m**power * rout_m ** (3.0 - power))
    )
    return rho0 / M_H2 / 1.0e6


# =============================================================================
# Flared / Pringle disk
# =============================================================================
def pringle_scale_height_au(
    r_au,
    rstar_rsun=10.0,
    height_fraction=0.01,
    flaring_power=1.25,
):
    r = np.asarray(r_au, dtype=float)
    rstar_au = rstar_rsun * R_SUN_TO_AU
    hstar_au = height_fraction * rstar_au
    return hstar_au * (r / rstar_au) ** flaring_power


def pringle_midplane_density_kg_m3(
    r_au,
    n0_cm3=1.0e7,
    r0_au=3600.0,
    density_power=-2.25,
):
    r = np.asarray(r_au, dtype=float)
    n_m3 = n0_cm3 * 1.0e6 * (r / r0_au) ** density_power
    return n_m3 * M_H2


def pringle_surface_density_kg_m2(
    r_au,
    n0_cm3,
    r0_au,
    density_power,
    rstar_rsun,
    height_fraction,
    flaring_power,
):
    rho_mid = pringle_midplane_density_kg_m3(
        r_au, n0_cm3=n0_cm3, r0_au=r0_au,
        density_power=density_power,
    )
    h_m = pringle_scale_height_au(
        r_au,
        rstar_rsun=rstar_rsun,
        height_fraction=height_fraction,
        flaring_power=flaring_power,
    ) * AU
    return np.sqrt(2.0 * np.pi) * rho_mid * h_m


def pringle_disk_mass_msun(
    rmax_au,
    n0_cm3,
    r0_au,
    density_power,
    rstar_rsun,
    height_fraction,
    flaring_power,
    nradial=2000,
):
    rstar_au = rstar_rsun * R_SUN_TO_AU
    r = np.geomspace(rstar_au, rmax_au, int(nradial))
    sigma = pringle_surface_density_kg_m2(
        r, n0_cm3, r0_au, density_power, rstar_rsun,
        height_fraction, flaring_power,
    )
    r_m = r * AU
    integrand = 2.0 * np.pi * r_m * sigma
    return float(np.trapz(integrand, r_m) / M_SUN)


def _ring_radial_accel_per_kg(field_r_m, ring_r_m, z_m):
    """Radial acceleration per kg of a circular ring."""
    R = float(field_r_m)
    a = np.asarray(ring_r_m, dtype=float)
    z = np.asarray(z_m, dtype=float)

    d2 = (a + R) ** 2 + z**2
    d = np.sqrt(d2)
    m = 4.0 * a * R / d2
    m = np.clip(m, 1.0e-15, 1.0 - 1.0e-13)

    K = ellipk(m)
    E = ellipe(m)
    dlnm_dR = 1.0 / R - 2.0 * (a + R) / d2
    dK_dR = (E / (2.0 * (1.0 - m)) - 0.5 * K) * dlnm_dR
    d_K_over_D_dR = dK_dR / d - K * (a + R) / d**3
    return (2.0 * G / np.pi) * d_K_over_D_dR


def pringle_disk_radial_acceleration_m_s2(
    r_eval_au,
    n0_cm3,
    r0_au,
    density_power,
    rstar_rsun,
    height_fraction,
    flaring_power,
    rmax_au,
    nradial=700,
    nz=12,
):
    """Axisymmetric disk radial acceleration in the midplane."""
    r_eval = np.asarray(r_eval_au, dtype=float)
    rstar_au = rstar_rsun * R_SUN_TO_AU

    if np.any(r_eval <= 0):
        raise ValueError("Disk evaluation radii must be > 0")
    if rmax_au <= rstar_au:
        raise ValueError("disk rmax must be > Rstar")
    if nz % 2 != 0:
        raise ValueError("disk nz must be even")

    a_au = np.geomspace(rstar_au, rmax_au, int(nradial))
    a_m = a_au * AU
    rho_mid = pringle_midplane_density_kg_m3(
        a_au, n0_cm3=n0_cm3, r0_au=r0_au,
        density_power=density_power,
    )
    h_m = pringle_scale_height_au(
        a_au,
        rstar_rsun=rstar_rsun,
        height_fraction=height_fraction,
        flaring_power=flaring_power,
    ) * AU

    xh, wh = np.polynomial.hermite.hermgauss(int(nz))
    z_m = np.sqrt(2.0) * h_m[:, None] * xh[None, :]

    dm_da = (
        2.0 * np.pi * a_m[:, None] * rho_mid[:, None]
        * np.sqrt(2.0) * h_m[:, None] * wh[None, :]
    )

    g = np.empty_like(r_eval)
    for j, R_au in enumerate(r_eval):
        g_per_kg = _ring_radial_accel_per_kg(
            R_au * AU, a_m[:, None], z_m
        )
        integrand = np.sum(g_per_kg * dm_da, axis=1)
        g[j] = np.trapz(integrand, a_m)
    return g


def precompute_disk_kernel(r_eval_au, disk_cfg, rmax_source_au):
    """Precompute the geometry-only kernel for a flared axisymmetric disk.

    The vertical structure H(R) is fixed, but diskPower and Rdisk may vary.
    For a given midplane-density shape rho_shape(a), the acceleration is

        g_R(R) = integral K(R,a) rho_shape(a) da.

    We also precompute the geometrical mass integrand so the same arbitrary
    density normalization can be converted to a requested total Mdisk.
    """
    rstar_rsun = float(disk_cfg.get("rstar_rsun", 10.0))
    rstar_au = rstar_rsun * R_SUN_TO_AU
    r0_au = float(disk_cfg.get("r0_au", 3600.0))
    height_fraction = float(disk_cfg.get("height_fraction", 0.01))
    flaring_power = float(disk_cfg.get("flaring_power", 1.25))
    nradial = int(disk_cfg.get("nradial", 700))
    nz = int(disk_cfg.get("nz", 12))

    if rmax_source_au <= rstar_au:
        raise ValueError("Maximum Rdisk prior must be > Rstar")
    if nradial < 100:
        raise ValueError("disk.nradial must be >= 100")
    if nz < 4 or nz % 2 != 0:
        raise ValueError("disk.nz must be even and >= 4")

    a_au = np.geomspace(rstar_au, float(rmax_source_au), nradial)
    a_m = a_au * AU
    h_m = pringle_scale_height_au(
        a_au,
        rstar_rsun=rstar_rsun,
        height_fraction=height_fraction,
        flaring_power=flaring_power,
    ) * AU

    # Vertical Gaussian quadrature. These positions/weights do not depend on
    # diskPower or Rdisk, so all elliptic-integral work can be cached here.
    xh, wh = np.polynomial.hermite.hermgauss(nz)
    z_m = np.sqrt(2.0) * h_m[:, None] * xh[None, :]

    # Geometry multiplying the midplane density rho_mid(a):
    # dM/da = rho_mid * mass_geom.
    vertical_geom = (
        2.0 * np.pi * a_m[:, None] * np.sqrt(2.0)
        * h_m[:, None] * wh[None, :]
    )
    mass_geom = 2.0 * np.pi * a_m * np.sqrt(2.0 * np.pi) * h_m

    r_eval = np.asarray(r_eval_au, dtype=float)
    kernel = np.empty((len(r_eval), nradial), dtype=float)

    print("Precomputing disk gravity kernel...")
    t0 = time.time()
    for j, R_au in enumerate(r_eval):
        g_per_kg = _ring_radial_accel_per_kg(
            float(R_au) * AU, a_m[:, None], z_m
        )
        kernel[j] = np.sum(g_per_kg * vertical_geom, axis=1)

    print(
        f"  kernel: {len(r_eval)} field radii x {nradial} source radii "
        f"x {nz} vertical nodes"
    )
    print(f"  disk-kernel precomputation took {time.time() - t0:.2f} s")

    return {
        "a_au": a_au,
        "a_m": a_m,
        "kernel": kernel,
        "mass_geom": mass_geom,
        "r0_au": r0_au,
        "rstar_au": rstar_au,
    }


def _integration_weights_to_limit(x, xmax):
    """Trapezoidal integration weights from x[0] to an arbitrary xmax.

    The final, partially covered interval is integrated using a linear
    interpolation of the integrand, avoiding a staircase likelihood in Rdisk.
    """
    x = np.asarray(x, dtype=float)
    xmax = float(xmax)
    if xmax <= x[0] or xmax > x[-1] * (1.0 + 1e-12):
        return None
    xmax = min(xmax, x[-1])

    w = np.zeros_like(x)
    i = int(np.searchsorted(x, xmax, side="right") - 1)

    # Complete intervals [x[k], x[k+1]] for k < i.
    if i > 0:
        dx = np.diff(x[: i + 1])
        w[:i] += 0.5 * dx
        w[1 : i + 1] += 0.5 * dx

    # Partial interval [x[i], xmax], unless xmax is exactly x[i].
    if i < len(x) - 1 and xmax > x[i]:
        dx_full = x[i + 1] - x[i]
        dx_part = xmax - x[i]
        t = dx_part / dx_full
        # Integral of linearly interpolated f over the partial interval.
        w[i] += 0.5 * dx_part * (2.0 - t)
        w[i + 1] += 0.5 * dx_part * t

    return w


def disk_acceleration_per_msun_from_shape(
    disk_kernel, density_power, rdisk_au
):
    """Return g_R(R) for a 1-Msun disk of the requested shape.

    The midplane density is proportional to (R/r0)^density_power and the disk
    is truncated at rdisk_au. The arbitrary density amplitude cancels when we
    normalize the resulting acceleration to one solar mass.
    """
    a_au = disk_kernel["a_au"]
    a_m = disk_kernel["a_m"]
    r0_au = disk_kernel["r0_au"]

    weights = _integration_weights_to_limit(a_m, float(rdisk_au) * AU)
    if weights is None:
        return None

    # Dimensionless density shape; amplitude is intentionally arbitrary.
    rho_shape = (a_au / r0_au) ** float(density_power)
    weighted_shape = rho_shape * weights

    # mass_geom * rho_shape integrated over da. Its dimensions correspond to
    # the mass produced by an arbitrary rho normalization of 1 kg/m^3.
    mass_shape_kg = float(
        np.dot(disk_kernel["mass_geom"], weighted_shape)
    )
    if not np.isfinite(mass_shape_kg) or mass_shape_kg <= 0:
        return None

    g_shape = disk_kernel["kernel"] @ weighted_shape
    return g_shape * (M_SUN / mass_shape_kg)


def disk_n0_cm3_from_total_mass(
    total_mass_msun, density_power, rdisk_au, disk_cfg
):
    """Midplane H2 density n(r0) implied by Mdisk for a given disk shape."""
    template_n0 = 1.0e7
    template_mass = pringle_disk_mass_msun(
        rmax_au=float(rdisk_au),
        n0_cm3=template_n0,
        r0_au=float(disk_cfg.get("r0_au", 3600.0)),
        density_power=float(density_power),
        rstar_rsun=float(disk_cfg.get("rstar_rsun", 10.0)),
        height_fraction=float(disk_cfg.get("height_fraction", 0.01)),
        flaring_power=float(disk_cfg.get("flaring_power", 1.25)),
        nradial=max(1500, int(disk_cfg.get("nradial", 700))),
    )
    return template_n0 * float(total_mass_msun) / template_mass


# =============================================================================
# Global data used by emcee workers
# =============================================================================
OBS_R_AU = None
OBS_V_KMS = None
OBS_SIGNS = None
SIGMA_V_KMS = None
DISK_KERNEL = None
ENV_ROUT_AU = None
PARAM_NAMES = None
PARAM_MIN = None
PARAM_MAX = None
FIXED_PARAMS = None


def model_velocity_signed(params):
    """Return model signed LOS velocities at the observed radii."""
    pdict = FIXED_PARAMS.copy()
    pdict.update(zip(PARAM_NAMES, params))

    mstar = float(pdict["MStar"])
    mdisk = float(pdict["Mdisk"])
    menv = float(pdict["Menv"])
    disk_power = float(pdict["diskPower"])
    rdisk = float(pdict["Rdisk"])
    env_power = float(pdict["envPower"])
    incl = float(pdict["incl"])

    if mstar < 0 or mdisk < 0 or menv < 0:
        return None
    if not (0.0 < incl <= 90.0):
        return None
    if env_power >= 3.0:
        return None

    gdisk_per_msun = disk_acceleration_per_msun_from_shape(
        DISK_KERNEL, disk_power, rdisk
    )
    if gdisk_per_msun is None or np.any(~np.isfinite(gdisk_per_msun)):
        return None

    r_m = OBS_R_AU * AU

    # Point mass
    v2 = G * mstar * M_SUN / r_m

    # Spherical envelope
    menclosed = envelope_enclosed_mass_msun(
        OBS_R_AU,
        total_mass_msun=menv,
        power=env_power,
        rout_au=ENV_ROUT_AU,
    )
    v2 += G * menclosed * M_SUN / r_m

    # Axisymmetric disk. g_R is signed: negative=inward, positive=outward.
    gdisk = gdisk_per_msun * mdisk
    v2 += -r_m * gdisk

    if np.any(~np.isfinite(v2)) or np.any(v2 <= 0):
        return None

    sini = np.sin(np.deg2rad(incl))
    vmag = np.sqrt(v2) / 1000.0 * sini
    return OBS_SIGNS * vmag


def ln_likelihood(params):
    # Uniform priors from CSV
    if np.any(params <= PARAM_MIN) or np.any(params >= PARAM_MAX):
        return -np.inf

    model = model_velocity_signed(params)
    if model is None or np.any(~np.isfinite(model)):
        return -np.inf

    resid = OBS_V_KMS - model
    lnx2 = -0.5 * np.sum((resid / SIGMA_V_KMS) ** 2)
    return lnx2 if np.isfinite(lnx2) else -np.inf


# =============================================================================
# MCMC / diagnostics
# =============================================================================
def initialize_walkers(mean, pmin, pmax, nwalkers, frac_stddev, seed=None):
    rng = np.random.default_rng(seed)
    mean = np.asarray(mean, dtype=float)
    pmin = np.asarray(pmin, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    width = pmax - pmin
    sigma = frac_stddev * width

    p0 = np.empty((nwalkers, len(mean)), dtype=float)
    for w in range(nwalkers):
        for j in range(len(mean)):
            # Rejection sampling keeps every walker strictly inside the prior.
            for _ in range(10000):
                x = rng.normal(mean[j], sigma[j])
                if pmin[j] < x < pmax[j]:
                    p0[w, j] = x
                    break
            else:
                p0[w, j] = rng.uniform(pmin[j], pmax[j])
    return p0


def run_mcmc(p0, nwalkers, nburn, nsteps, npars, nthreads=1):
    pool = None
    try:
        if nthreads is not None and nthreads > 1:
            pool = Pool(processes=nthreads)

        sampler = emcee.EnsembleSampler(
            nwalkers, npars, ln_likelihood, pool=pool
        )

        state = p0
        if nburn > 0:
            print(f"\nRunning burn-in: {nburn} steps")
            t0 = time.time()
            state = sampler.run_mcmc(state, nburn, progress=True)
            print(f"Burn-in took {time.time() - t0:.1f} s")
            sampler.reset()

        print(f"\nRunning production: {nsteps} steps")
        t0 = time.time()
        sampler.run_mcmc(state, nsteps, progress=True)
        print(f"Production took {time.time() - t0:.1f} s")
        return sampler

    finally:
        if pool is not None:
            pool.close()
            pool.join()


def percentile_summary(samples, names):
    out = {}
    for j, name in enumerate(names):
        q16, q50, q84 = np.percentile(samples[:, j], [16, 50, 84])
        out[name] = {
            "median": float(q50),
            "minus": float(q50 - q16),
            "plus": float(q84 - q50),
        }
    return out


def save_diagnostics(sampler, param_names, p0_mean, outdir, tag):
    chain = sampler.get_chain()  # (steps, walkers, pars)
    samples = sampler.get_chain(flat=True)
    summary = percentile_summary(samples, param_names)
    best = np.array([summary[n]["median"] for n in param_names])

    np.savetxt(
        outdir / f"{tag}_parameter_samples.txt",
        samples,
        fmt="%.8e",
        header=" ".join(param_names),
    )

    # Same plot_helpers the user's other MCMC uses.
    if HAVE_PLOT_HELPERS:
        chain_legacy = np.transpose(chain, (1, 0, 2))  # walkers,steps,pars

        plot_walkers(chain_legacy.T, best, header=param_names)
        plt.tight_layout()
        plt.savefig(outdir / f"{tag}_walkers.png", dpi=300)
        plt.close()

        plot_corner(samples, labels=param_names)
        plt.savefig(outdir / f"{tag}_corner.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("WARNING: plot_helpers not found; skipping walkers/corner plots.")

    return samples, summary, best


# =============================================================================
# Best-fit plot
# =============================================================================
def evaluate_components(
    r_au, pdict, disk_kernel, env_rout
):
    r = np.asarray(r_au, dtype=float)
    r_m = r * AU
    mstar = float(pdict["MStar"])
    mdisk = float(pdict["Mdisk"])
    menv = float(pdict["Menv"])
    disk_power = float(pdict["diskPower"])
    rdisk = float(pdict["Rdisk"])
    env_power = float(pdict["envPower"])
    incl = float(pdict["incl"])
    sini = np.sin(np.deg2rad(incl))

    gdisk_per_msun = disk_acceleration_per_msun_from_shape(
        disk_kernel, disk_power, rdisk
    )
    if gdisk_per_msun is None:
        raise RuntimeError("Invalid best-fit disk shape")

    v2_star = G * mstar * M_SUN / r_m
    menclosed = envelope_enclosed_mass_msun(r, menv, env_power, env_rout)
    v2_env = G * menclosed * M_SUN / r_m
    v2_disk = -r_m * (gdisk_per_msun * mdisk)

    def vv(v2):
        out = np.full_like(v2, np.nan, dtype=float)
        ok = v2 > 0
        out[ok] = np.sqrt(v2[ok]) / 1000.0 * sini
        return out

    return {
        "star": vv(v2_star),
        "star_disk": vv(v2_star + v2_disk),
        "star_env": vv(v2_star + v2_env),
        "total": vv(v2_star + v2_env + v2_disk),
    }


def make_bestfit_plot(
    r_signed_all,
    v_all,
    excluded,
    fit_mask,
    sides,
    velocity_axis_rel,
    best_param_dict,
    disk_cfg,
    env_cfg,
    outpath,
    nsigma,
):
    rfit = np.abs(r_signed_all[fit_mask])
    rmin = np.nanmin(rfit)
    rmax = np.nanmax(np.abs(r_signed_all[np.isfinite(r_signed_all)]))
    rgrid = np.geomspace(rmin, rmax, 500)

    # One geometry-only disk kernel on the plotting grid.
    # Use the best-fit Rdisk itself as source-grid outer edge.
    plot_kernel = precompute_disk_kernel(
        rgrid, disk_cfg, rmax_source_au=float(best_param_dict["Rdisk"])
    )
    comp = evaluate_components(
        rgrid,
        best_param_dict,
        plot_kernel,
        float(env_cfg.get("rout_au", 3600.0)),
    )

    def median_sign(values, mask, default):
        if np.any(mask):
            med = np.nanmedian(values[mask])
            if np.isfinite(med) and med != 0:
                return float(np.sign(med))
        return default

    left = sides == -1
    right = sides == +1
    left_vsign = median_sign(v_all, left & fit_mask, +1.0)
    right_vsign = median_sign(v_all, right & fit_mask, -1.0)
    left_rsign = median_sign(r_signed_all, left, -1.0)
    right_rsign = median_sign(r_signed_all, right, +1.0)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    colors = {
        "data": "tab:blue",
        "excluded": "tab:gray",
        "star": "tab:orange",
        "star_disk": "tab:purple",
        "star_env": "tab:red",
        "total": "tab:brown",
    }

    ax.scatter(
        r_signed_all[~excluded], v_all[~excluded],
        s=28, color=colors["data"],
        label=rf"Observed edge $\geq {nsigma:g}\sigma$",
    )
    if np.any(excluded):
        ax.scatter(
            r_signed_all[excluded], v_all[excluded],
            s=50, facecolors="none", edgecolors=colors["excluded"],
            linewidths=1.2, label="Excluded",
        )

    labels = {
        "star": rf"$M_\star={best_param_dict['MStar']:.2f}\,M_\odot$",
        "star_disk": (
            rf"$M_\star+M_d$ ($M_d={best_param_dict['Mdisk']:.2f}\,M_\odot$, "
            rf"$q_d={best_param_dict['diskPower']:.2f}$, "
            rf"$R_d={best_param_dict['Rdisk']:.0f}$ AU)"
        ),
        "star_env": (
            rf"$M_\star+M_e$ ($M_e={best_param_dict['Menv']:.2f}\,M_\odot$, "
            rf"$p_e={best_param_dict['envPower']:.2f}$)"
        ),
        "total": rf"Total, $i={best_param_dict['incl']:.1f}^\circ$",
    }
    styles = {
        "star": "--",
        "star_disk": "-.",
        "star_env": ":",
        "total": "-",
    }

    for key in ["star", "star_disk", "star_env", "total"]:
        lw = 2.5 if key == "total" else 1.8
        ax.plot(
            left_rsign * rgrid,
            left_vsign * comp[key],
            color=colors[key], ls=styles[key], lw=lw,
            label=labels[key],
        )
        ax.plot(
            right_rsign * rgrid,
            right_vsign * comp[key],
            color=colors[key], ls=styles[key], lw=lw,
        )

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
    ax.axvline(0, color="black", lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("Projected spatial offset [AU]")
    ax.set_ylabel(r"$v-v_{\rm sys}$ [km/s]")
    ax.grid(alpha=0.25)

    # Same fixed 1.5 x PV velocity range used previously.
    vmin = np.nanmin(velocity_axis_rel)
    vmax = np.nanmax(velocity_axis_rel)
    vc = 0.5 * (vmin + vmax)
    dv = vmax - vmin
    ax.set_ylim(vc - 0.75 * dv, vc + 0.75 * dv)

    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MCMC fit of central mass + Pringle disk + spherical envelope to a PV rotation curve."
    )
    parser.add_argument(
        "config", nargs="?", default="configs/G328_rotation.json",
        help="JSON configuration file",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = root_dir / config_file

    with open(config_file) as f:
        config = json.load(f)

    obs_cfg = config["observation"]
    model_cfg = config["model"]
    disk_cfg = model_cfg["disk"]
    env_cfg = model_cfg["envelope"]
    mcmc_cfg = config["mcmc"]
    output_cfg = config.get("output", {})

    # Avoid hidden OpenMP over-subscription when multiprocessing is used.
    os.environ["OMP_NUM_THREADS"] = str(mcmc_cfg.get("omp_threads", 1))

    # ------------------------------------------------------------------
    # Read PV and extract observed curve
    # ------------------------------------------------------------------
    pv_file = Path(obs_cfg["pv_file"])
    if not pv_file.is_absolute():
        pv_file = root_dir / "../pv" / pv_file

    with fits.open(pv_file) as hdul:
        header = hdul[0].header.copy()
        data = np.squeeze(np.asarray(hdul[0].data, dtype=float))

    if data.ndim != 2:
        raise ValueError(f"PV must be 2-D after squeeze; got {data.shape}")

    nvel, nspace = data.shape
    spatial_arcsec = spatial_axis_arcsec(header, nspace)
    velocity_abs = velocity_axis_kms(
        header, nvel, rest_freq_ghz=obs_cfg.get("rest_freq_ghz")
    )

    center_arcsec = float(obs_cfg.get("center_arcsec", 0.0))
    center_col = int(np.argmin(np.abs(spatial_arcsec - center_arcsec)))

    vsys_arg = float(obs_cfg.get("vsys_kms", 0.0))
    if np.isclose(vsys_arg, 0.0):
        vsys = spectral_center_velocity(velocity_abs)
        vsys_mode = "PV spectral center"
    else:
        vsys = vsys_arg
        vsys_mode = "manual"
    velocity_rel = velocity_abs - vsys

    sigma_flux = float(obs_cfg.get("sigma_flux", 2.3e-3))
    nsigma = float(obs_cfg.get("nsigma", 5.0))
    threshold = nsigma * sigma_flux

    rows, cols, sides = extract_envelope(
        data, threshold, center_col, velocity_rel
    )
    if len(cols) == 0:
        raise RuntimeError("No points found above threshold")

    distance_pc = float(obs_cfg.get("distance_pc", 2500.0))
    offset_arcsec = spatial_arcsec - center_arcsec
    r_signed_au = offset_arcsec[cols] * distance_pc
    r_abs_au = np.abs(r_signed_au)
    v_pts = velocity_rel[rows]

    discard = parse_ranges(obs_cfg.get("discard", []))
    excluded = columns_in_ranges(cols, discard)
    fit_mask = (
        (~excluded)
        & np.isfinite(r_abs_au)
        & np.isfinite(v_pts)
        & (r_abs_au > 0)
    )
    if np.count_nonzero(fit_mask) < 4:
        raise RuntimeError("Fewer than 4 points remain for MCMC")

    # Branch signs used by the symmetric dynamical model.
    def median_sign(values, mask, default):
        if np.any(mask):
            med = np.nanmedian(values[mask])
            if np.isfinite(med) and med != 0:
                return float(np.sign(med))
        return default

    left_vsign = median_sign(v_pts, (sides == -1) & fit_mask, +1.0)
    right_vsign = median_sign(v_pts, (sides == +1) & fit_mask, -1.0)
    obs_signs = np.where(sides[fit_mask] == -1, left_vsign, right_vsign)

    # Velocity uncertainty in likelihood. If omitted, use one spectral channel.
    sigma_v = mcmc_cfg.get("noise_kms")
    if sigma_v is None:
        sigma_v = float(np.nanmedian(np.abs(np.diff(velocity_rel))))
        print(f"noise_kms not set: using one channel = {sigma_v:.4f} km/s")
    sigma_v = float(sigma_v)
    if sigma_v <= 0:
        raise ValueError("mcmc.noise_kms must be > 0")

    # ------------------------------------------------------------------
    # Parameter CSV
    # ------------------------------------------------------------------
    parameter_file = Path(mcmc_cfg["parameter_file"])
    if not parameter_file.is_absolute():
        parameter_file = root_dir / "configs" / parameter_file

    param_space = pd.read_csv(parameter_file, skipinitialspace=True)
    param_space.set_index("Parameter", inplace=True)

    required = {
        "MStar", "Mdisk", "Menv", "diskPower", "Rdisk", "envPower", "incl"
    }
    missing = required - set(param_space.index)
    if missing:
        raise ValueError(f"Missing required parameters in CSV: {sorted(missing)}")

    free = param_space[param_space["Fit"].astype(bool)]
    fixed = param_space[~param_space["Fit"].astype(bool)]

    param_names = free.index.tolist()
    p0_mean = free["Mean"].to_numpy(dtype=float)
    param_min = free["Min"].to_numpy(dtype=float)
    param_max = free["Max"].to_numpy(dtype=float)
    fixed_params = fixed["Mean"].astype(float).to_dict()

    npars = len(param_names)
    nwalkers = int(mcmc_cfg.get("nwalkers", max(16, 4 * npars)))
    nburn = int(mcmc_cfg.get("nburn", 300))
    nsteps = int(mcmc_cfg.get("nsteps", 1500))
    frac_stddev = float(mcmc_cfg.get("frac_stddev", 0.03))
    nthreads = int(mcmc_cfg.get("nthreads", 1))
    seed = mcmc_cfg.get("seed")

    if nwalkers < 2 * npars:
        raise ValueError("emcee requires at least ~2*npars walkers; use more.")

    # ------------------------------------------------------------------
    # Precompute the geometry-only disk kernel at MCMC radii.
    # diskPower and Rdisk remain free; no elliptic integrals are evaluated
    # inside the likelihood.
    # ------------------------------------------------------------------
    r_fit = r_abs_au[fit_mask]

    if "Rdisk" in param_names:
        rdisk_prior_max = float(param_max[param_names.index("Rdisk")])
    else:
        rdisk_prior_max = float(fixed_params["Rdisk"])

    disk_kernel_fit = precompute_disk_kernel(
        r_fit, disk_cfg, rmax_source_au=rdisk_prior_max
    )

    # ------------------------------------------------------------------
    # Populate globals for workers
    # ------------------------------------------------------------------
    global OBS_R_AU, OBS_V_KMS, OBS_SIGNS, SIGMA_V_KMS
    global DISK_KERNEL, ENV_ROUT_AU
    global PARAM_NAMES, PARAM_MIN, PARAM_MAX, FIXED_PARAMS

    OBS_R_AU = r_fit
    OBS_V_KMS = v_pts[fit_mask]
    OBS_SIGNS = obs_signs
    SIGMA_V_KMS = sigma_v
    DISK_KERNEL = disk_kernel_fit
    ENV_ROUT_AU = float(env_cfg.get("rout_au", 3600.0))
    PARAM_NAMES = param_names
    PARAM_MIN = param_min
    PARAM_MAX = param_max
    FIXED_PARAMS = fixed_params

    # Validate fixed parameters by building a full parameter dictionary.
    means_all = param_space["Mean"].astype(float).to_dict()
    for key in required:
        if key not in means_all:
            raise ValueError(f"Missing {key}")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    tag = output_cfg.get("tag", "rotation_mcmc")
    outdir = Path(output_cfg.get("dir", f"{tag}_results"))
    if not outdir.is_absolute():
        outdir = root_dir / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n-------------- INPUT DATA --------------")
    print(f"PV: {pv_file}")
    print(f"shape: {data.shape}")
    print(f"v_sys = {vsys:.5f} km/s ({vsys_mode})")
    print(f"threshold = {threshold:.4g} = {nsigma:g} sigma")
    print(f"extracted points = {len(cols)}")
    print(f"excluded points = {np.count_nonzero(excluded)}")
    print(f"MCMC points = {np.count_nonzero(fit_mask)}")
    print(f"sigma_v = {sigma_v:.4f} km/s")
    print(f"free parameters = {param_names}")
    print("----------------------------------------\n")

    p0 = initialize_walkers(
        p0_mean, param_min, param_max,
        nwalkers=nwalkers,
        frac_stddev=frac_stddev,
        seed=seed,
    )

    sampler = run_mcmc(
        p0, nwalkers, nburn, nsteps, npars, nthreads=nthreads
    )

    samples, summary, best_free = save_diagnostics(
        sampler, param_names, p0_mean, outdir, tag
    )

    best_params = fixed_params.copy()
    best_params.update(zip(param_names, best_free))

    # Derived posterior ratios when relevant parameters are sampled/fixed.
    # Build full sample arrays for all four physical parameters.
    nsamp = len(samples)
    full_samples = {}
    for name in required:
        if name in param_names:
            full_samples[name] = samples[:, param_names.index(name)]
        else:
            full_samples[name] = np.full(nsamp, float(fixed_params[name]))

    qdisk = full_samples["Mdisk"] / full_samples["MStar"]
    qenv = full_samples["Menv"] / full_samples["MStar"]

    qd16, qd50, qd84 = np.percentile(qdisk, [16, 50, 84])
    qe16, qe50, qe84 = np.percentile(qenv, [16, 50, 84])

    # Density normalizations implied by the median masses and median shapes.
    disk_n0_best = disk_n0_cm3_from_total_mass(
        best_params["Mdisk"],
        density_power=best_params["diskPower"],
        rdisk_au=best_params["Rdisk"],
        disk_cfg=disk_cfg,
    )
    env_n0_best = envelope_n0_cm3_from_total_mass(
        best_params["Menv"],
        power=best_params["envPower"],
        rout_au=ENV_ROUT_AU,
        r0_au=float(env_cfg.get("r0_au", 3600.0)),
    )

    result = {
        "free_parameter_summary": summary,
        "best_parameters": {k: float(v) for k, v in best_params.items()},
        "derived": {
            "Mdisk_over_Mstar": {
                "median": float(qd50),
                "minus": float(qd50 - qd16),
                "plus": float(qd84 - qd50),
            },
            "Menv_over_Mstar": {
                "median": float(qe50),
                "minus": float(qe50 - qe16),
                "plus": float(qe84 - qe50),
            },
            "disk_n0_cm3_at_r0": float(disk_n0_best),
            "envelope_n0_cm3_at_r0": float(env_n0_best),
            "disk_power": float(best_params["diskPower"]),
            "disk_rout_au": float(best_params["Rdisk"]),
            "envelope_power": float(best_params["envPower"]),
        },
        "data": {
            "vsys_kms": float(vsys),
            "sigma_v_kms": float(sigma_v),
            "n_points_used": int(np.count_nonzero(fit_mask)),
        },
    }

    with open(outdir / f"{tag}_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    make_bestfit_plot(
        r_signed_all=r_signed_au,
        v_all=v_pts,
        excluded=excluded,
        fit_mask=fit_mask,
        sides=sides,
        velocity_axis_rel=velocity_rel,
        best_param_dict=best_params,
        disk_cfg=disk_cfg,
        env_cfg=env_cfg,
        outpath=outdir / f"{tag}_bestfit.png",
        nsigma=nsigma,
    )

    print("\n================ RESULTS ================")
    for name in [
        "MStar", "Mdisk", "Menv", "diskPower", "Rdisk", "envPower", "incl"
    ]:
        if name in summary:
            s = summary[name]
            print(
                f"{name:>8s} = {s['median']:.5g} "
                f"-{s['minus']:.3g} +{s['plus']:.3g}"
            )
        else:
            print(f"{name:>8s} = {best_params[name]:.5g} (fixed)")

    print(
        f"Mdisk/MStar = {qd50:.4g} -{qd50-qd16:.3g} +{qd84-qd50:.3g}"
    )
    print(
        f"Menv/MStar  = {qe50:.4g} -{qe50-qe16:.3g} +{qe84-qe50:.3g}"
    )
    print(f"Implied disk n0 = {disk_n0_best:.4g} cm^-3")
    print(f"Implied env  n0 = {env_n0_best:.4g} cm^-3")
    print(f"Outputs: {outdir}")
    print("=========================================\n")


if __name__ == "__main__":
    main()
