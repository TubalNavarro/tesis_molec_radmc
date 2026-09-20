#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.special import ellipk, ellipe

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------
G = 6.67430e-11               # m^3 kg^-1 s^-2
M_SUN = 1.98847e30            # kg
M_H = 1.6735575e-27           # kg
M_H2 = 2.0 * M_H              # kg; cada partícula de la envolvente es H2
AU = 1.495978707e11           # m
R_SUN = 6.957e8               # m
R_SUN_TO_AU = R_SUN / AU
C_KMS = 299792.458            # km/s
PC_TO_AU = 206264.80624709636
V_1AU_1MSUN = np.sqrt(G * M_SUN / AU) / 1000.0  # 29.7847 km/s


# -----------------------------------------------------------------------------
# Ejes FITS
# -----------------------------------------------------------------------------
def linear_axis_from_header(header, fits_axis: int, n: int) -> np.ndarray:
    """Construye un eje lineal FITS a partir de CRVAL, CRPIX y CDELT."""
    crval = float(header.get(f"CRVAL{fits_axis}", 0.0))
    crpix = float(header.get(f"CRPIX{fits_axis}", 1.0))

    cdelt = header.get(f"CDELT{fits_axis}")
    if cdelt is None:
        cdelt = header.get(f"CD{fits_axis}_{fits_axis}")
    if cdelt is None:
        raise KeyError(
            f"No encuentro CDELT{fits_axis} ni CD{fits_axis}_{fits_axis}."
        )

    pix_fits = np.arange(n, dtype=float) + 1.0  # FITS es 1-based
    return crval + (pix_fits - crpix) * float(cdelt)


def spatial_axis_arcsec(header, ncols: int) -> np.ndarray:
    """Devuelve el eje espacial FITS 1 en arcsec."""
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


def velocity_axis_kms(
    header,
    nrows: int,
    rest_freq_ghz: float | None = None,
) -> np.ndarray:
    """Devuelve el eje FITS 2 como velocidad [km/s].

    Si CTYPE2=FREQ usa la convención radio:
        v = c * (nu0 - nu) / nu0
    """
    y = linear_axis_from_header(header, 2, nrows)
    unit = str(header.get("CUNIT2", "")).strip().lower().replace(" ", "")
    ctype = str(header.get("CTYPE2", "")).upper()

    # Eje ya en velocidad
    if "VELO" in ctype or "VRAD" in ctype:
        if unit in {"km/s", "kms-1", "km.s-1"}:
            return y
        return y / 1000.0

    # Eje en frecuencia
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
            nu0_hz = rest_freq_ghz * 1e9
        else:
            nu0_hz = header.get("RESTFRQ", header.get("RESTFREQ"))
            if nu0_hz is None:
                raise ValueError(
                    "No encuentro RESTFRQ/RESTFREQ. Usa --rest-freq-ghz."
                )
            nu0_hz = float(nu0_hz)

        return C_KMS * (nu0_hz - nu_hz) / nu0_hz

    raise ValueError(
        f"No puedo interpretar el eje 2: CTYPE2='{ctype}', CUNIT2='{unit}'."
    )


def spectral_center_velocity(velocity_kms: np.ndarray) -> float:
    """Velocidad en el centro geométrico del eje espectral."""
    center_row = (len(velocity_kms) - 1) / 2.0
    return float(
        np.interp(
            center_row,
            np.arange(len(velocity_kms), dtype=float),
            velocity_kms,
        )
    )


def distance_to_au(distance: float, unit: str) -> float:
    if unit == "pc":
        return distance * PC_TO_AU
    if unit == "kpc":
        return distance * 1e3 * PC_TO_AU
    if unit == "au":
        return distance
    raise ValueError("distance-unit debe ser pc, kpc o au.")


# -----------------------------------------------------------------------------
# Selección de columnas
# -----------------------------------------------------------------------------
def parse_ranges(items: list[str] | None) -> list[tuple[int, int]]:
    """Convierte --discard 55:70 80:82 a rangos inclusivos 0-based."""
    if not items:
        return []

    ranges = []
    for item in items:
        try:
            lo, hi = map(int, item.split(":", 1))
        except Exception as exc:
            raise ValueError(
                f"Rango inválido '{item}'. Usa, por ejemplo, 55:70"
            ) from exc

        if lo > hi:
            lo, hi = hi, lo
        ranges.append((lo, hi))

    return ranges


def columns_in_ranges(
    cols: np.ndarray,
    ranges: list[tuple[int, int]],
) -> np.ndarray:
    mask = np.zeros(cols.shape, dtype=bool)
    for lo, hi in ranges:
        mask |= (cols >= lo) & (cols <= hi)
    return mask


# -----------------------------------------------------------------------------
# Extracción de la envolvente del PV
# -----------------------------------------------------------------------------
def extract_envelope(
    data: np.ndarray,
    threshold: float,
    center_col: int,
    velocity_rel_kms: np.ndarray,
):
    """Extrae la envolvente siguiendo el sentido visual pedido.

    Para cada columna espacial:

    * Mitad izquierda: busca desde ARRIBA hacia ABAJO.
      En una gráfica v vs r, arriba corresponde al mayor valor de velocidad.

    * Mitad derecha: busca desde ABAJO hacia ARRIBA.
      Abajo corresponde al menor valor de velocidad.

    Se selecciona el PRIMER píxel >= threshold encontrado en ese recorrido.
    El orden se construye usando el eje físico de velocidad, así que funciona
    independientemente del signo de CDELT2.
    """
    nvel, nspace = data.shape
    rows, cols, sides = [], [], []

    if len(velocity_rel_kms) != nvel:
        raise ValueError("El eje de velocidad no coincide con el número de filas del PV.")

    finite_vel = np.isfinite(velocity_rel_kms)

    # top -> bottom = velocidad mayor -> velocidad menor
    top_to_bottom = np.argsort(
        np.where(finite_vel, velocity_rel_kms, -np.inf)
    )[::-1]

    # bottom -> top = velocidad menor -> velocidad mayor
    bottom_to_top = np.argsort(
        np.where(finite_vel, velocity_rel_kms, np.inf)
    )

    # Mitad izquierda: ARRIBA -> ABAJO
    for col in range(center_col):
        for row in top_to_bottom:
            if not finite_vel[row]:
                continue
            value = data[row, col]
            if np.isfinite(value) and value >= threshold:
                rows.append(int(row))
                cols.append(col)
                sides.append(-1)
                break

    # Mitad derecha: ABAJO -> ARRIBA
    for col in range(center_col + 1, nspace):
        for row in bottom_to_top:
            if not finite_vel[row]:
                continue
            value = data[row, col]
            if np.isfinite(value) and value >= threshold:
                rows.append(int(row))
                cols.append(col)
                sides.append(+1)
                break

    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(sides, dtype=int),
    )


# -----------------------------------------------------------------------------
# Kepler y envolvente esférica
# -----------------------------------------------------------------------------
def fit_keplerian_mass(
    r_au: np.ndarray,
    velocity_rel_kms: np.ndarray,
    incl_deg: float,
):
    """Ajusta |v-vsys| = A/sqrt(r) y devuelve M puntual [M_sun]."""
    sini = np.sin(np.deg2rad(incl_deg))
    if sini <= 0:
        raise ValueError("--incl debe ser > 0 grados.")

    x = 1.0 / np.sqrt(r_au)
    y = np.abs(velocity_rel_kms)

    amp = np.sum(x * y) / np.sum(x * x)
    model = amp * x
    residuals = y - model

    mass_msun = (amp / (V_1AU_1MSUN * sini)) ** 2

    if len(x) > 1 and amp > 0:
        variance = np.sum(residuals**2) / (len(x) - 1)
        sigma_amp = np.sqrt(variance / np.sum(x * x))
        sigma_mass = 2.0 * mass_msun * sigma_amp / amp
    else:
        sigma_mass = np.nan

    return mass_msun, sigma_mass, residuals


def envelope_mass_msun(
    r_au: np.ndarray | float,
    n0_cm3: float = 1.0e7,
    r0_au: float = 3600.0,
    density_power: float = 1.5,
    rout_au: float | None = None,
) -> np.ndarray:
    """Masa encerrada de una envolvente esférica n(r)=n0 (r/r0)^(-p).

    Cada partícula se toma como una molécula H2 de masa 2*m_H.

    Para p < 3, la integral desde r=0 es finita:

        M(<r) = 4*pi*rho0*r0^3/(3-p) * (r/r0)^(3-p)

    donde rho0 = n0 * m_H2.

    Si ``rout_au`` se especifica, para r > rout se mantiene constante
    M_env(<r) = M_env(<rout), como corresponde a una envolvente truncada.
    """
    if n0_cm3 <= 0:
        raise ValueError("La densidad --envelope-n0 debe ser > 0 cm^-3.")
    if r0_au <= 0:
        raise ValueError("--envelope-r0-au debe ser > 0.")
    if density_power >= 3.0:
        raise ValueError(
            "La integral desde r=0 diverge para density_power >= 3. "
            "Usa un perfil con p < 3."
        )
    if rout_au is not None and rout_au <= 0:
        raise ValueError("--envelope-rout-au debe ser > 0 si se especifica.")

    r = np.asarray(r_au, dtype=float)
    if np.any(r < 0):
        raise ValueError("Los radios deben ser >= 0 AU.")

    # Para una envolvente truncada, la masa encerrada deja de crecer en Rout.
    r_eff = np.minimum(r, rout_au) if rout_au is not None else r

    # n0 [cm^-3] -> [m^-3]
    n0_m3 = n0_cm3 * 1.0e6
    rho0 = n0_m3 * M_H2  # kg m^-3

    r0_m = r0_au * AU
    prefactor_kg = 4.0 * np.pi * rho0 * r0_m**3 / (3.0 - density_power)

    mass_kg = prefactor_kg * (r_eff / r0_au) ** (3.0 - density_power)
    return mass_kg / M_SUN


def rotation_velocity_point_plus_envelope_kms(
    r_au: np.ndarray,
    point_mass_msun: float,
    incl_deg: float,
    n0_cm3: float = 1.0e7,
    r0_au: float = 3600.0,
    density_power: float = 1.5,
    rout_au: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocidad circular LOS de M_puntual + envolvente esférica.

    Por el teorema de las capas, sólo contribuye M_env(<r):

        v_circ(r) = sqrt[G (M_* + M_env(<r)) / r]
        v_LOS = v_circ sin(i)

    Devuelve (v_los_kms, M_env_msun).
    """
    if point_mass_msun < 0:
        raise ValueError("La masa puntual debe ser >= 0 M_sun.")

    r = np.asarray(r_au, dtype=float)
    if np.any(r <= 0):
        raise ValueError("La velocidad de rotación requiere r > 0 AU.")

    sini = np.sin(np.deg2rad(incl_deg))
    if sini <= 0:
        raise ValueError("--incl debe ser > 0 grados.")

    menv = envelope_mass_msun(
        r,
        n0_cm3=n0_cm3,
        r0_au=r0_au,
        density_power=density_power,
        rout_au=rout_au,
    )
    mtot = point_mass_msun + menv
    v_los = V_1AU_1MSUN * np.sqrt(mtot / r) * sini
    return v_los, menv



# -----------------------------------------------------------------------------
# Disco de Pringle / disco flared axisimétrico
# -----------------------------------------------------------------------------
def pringle_scale_height_au(
    r_au: np.ndarray | float,
    rstar_rsun: float = 10.0,
    height_fraction: float = 0.01,
    flaring_power: float = 1.25,
) -> np.ndarray:
    """Escala de altura H(R) del disco.

    Se usa

        H(R) = H_* (R / R_*)^beta,
        H_*  = height_fraction * R_*.

    Con los defaults beta=1.25 y rho_mid ~ R^-2.25 se recupera la pareja
    habitual alpha=beta+1 del disco flared (alpha=2.25, beta=1.25).
    """
    if rstar_rsun <= 0:
        raise ValueError("--disk-rstar-rsun debe ser > 0.")
    if height_fraction <= 0:
        raise ValueError("--disk-height-fraction debe ser > 0.")
    if flaring_power <= 0:
        raise ValueError("--disk-flaring-power debe ser > 0.")

    r = np.asarray(r_au, dtype=float)
    if np.any(r <= 0):
        raise ValueError("Los radios del disco deben ser > 0 AU.")

    rstar_au = rstar_rsun * R_SUN_TO_AU
    hstar_au = height_fraction * rstar_au
    return hstar_au * (r / rstar_au) ** flaring_power


def pringle_midplane_density_kg_m3(
    r_au: np.ndarray | float,
    n0_cm3: float = 1.0e7,
    r0_au: float = 3600.0,
    density_power: float = -2.25,
) -> np.ndarray:
    """Densidad volumétrica en el plano medio del disco.

    La normalización se interpreta como densidad numérica de H2:

        n_mid(R) = n0 * (R/r0)^q

    con q=-2.25 por default. Cada partícula tiene masa 2 m_H.
    """
    if n0_cm3 <= 0:
        raise ValueError("--disk-n0 debe ser > 0 cm^-3.")
    if r0_au <= 0:
        raise ValueError("--disk-r0-au debe ser > 0.")

    r = np.asarray(r_au, dtype=float)
    if np.any(r <= 0):
        raise ValueError("Los radios del disco deben ser > 0 AU.")

    n_m3 = n0_cm3 * 1.0e6 * (r / r0_au) ** density_power
    return n_m3 * M_H2


def pringle_surface_density_kg_m2(
    r_au: np.ndarray | float,
    n0_cm3: float = 1.0e7,
    r0_au: float = 3600.0,
    density_power: float = -2.25,
    rstar_rsun: float = 10.0,
    height_fraction: float = 0.01,
    flaring_power: float = 1.25,
) -> np.ndarray:
    """Densidad superficial Sigma(R) para una vertical Gaussiana.

    rho(R,z) = rho_mid(R) exp[-z^2/(2 H(R)^2)]
    Sigma(R) = sqrt(2*pi) rho_mid(R) H(R)
    """
    r = np.asarray(r_au, dtype=float)
    rho_mid = pringle_midplane_density_kg_m3(
        r,
        n0_cm3=n0_cm3,
        r0_au=r0_au,
        density_power=density_power,
    )
    h_m = pringle_scale_height_au(
        r,
        rstar_rsun=rstar_rsun,
        height_fraction=height_fraction,
        flaring_power=flaring_power,
    ) * AU
    return np.sqrt(2.0 * np.pi) * rho_mid * h_m


def pringle_disk_mass_msun(
    rmax_au: float,
    n0_cm3: float = 1.0e7,
    r0_au: float = 3600.0,
    density_power: float = -2.25,
    rstar_rsun: float = 10.0,
    height_fraction: float = 0.01,
    flaring_power: float = 1.25,
    nradial: int = 2000,
) -> float:
    """Masa total del disco entre R_* y Rmax.

    Integra 2*pi*R*Sigma(R)dR. La integración vertical es analítica gracias
    a la Gaussiana.
    """
    rstar_au = rstar_rsun * R_SUN_TO_AU
    if rmax_au <= rstar_au:
        raise ValueError("--disk-rmax-au debe ser mayor que R_*.")
    if nradial < 50:
        raise ValueError("nradial debe ser >= 50.")

    r = np.geomspace(rstar_au, rmax_au, int(nradial))
    sigma = pringle_surface_density_kg_m2(
        r,
        n0_cm3=n0_cm3,
        r0_au=r0_au,
        density_power=density_power,
        rstar_rsun=rstar_rsun,
        height_fraction=height_fraction,
        flaring_power=flaring_power,
    )
    r_m = r * AU
    integrand = 2.0 * np.pi * r_m * sigma
    return float(np.trapz(integrand, r_m) / M_SUN)


def _ring_radial_accel_per_kg(
    field_r_m: float,
    ring_r_m: np.ndarray,
    z_m: np.ndarray,
) -> np.ndarray:
    """Aceleración radial por kg debida a anillos circulares.

    El potencial de un anillo de masa M y radio a, evaluado en (R,z), es

        Phi = -2 G M / (pi D) K(m),
        D^2 = (R+a)^2 + z^2,
        m   = 4 a R / D^2.

    Aquí se deriva analíticamente respecto a R para evitar diferencias
    finitas. El signo es radial: negativo=hacia el centro, positivo=hacia
    afuera. La salida es por unidad de masa del anillo.
    """
    R = float(field_r_m)
    if R <= 0:
        raise ValueError("El radio de evaluación debe ser > 0.")

    a = np.asarray(ring_r_m, dtype=float)
    z = np.asarray(z_m, dtype=float)

    d2 = (a + R) ** 2 + z**2
    d = np.sqrt(d2)
    m = 4.0 * a * R / d2

    # La altura finita evita m=1 exactamente. El clip sólo protege contra
    # redondeo numérico cuando R~a y |z| es muy pequeño.
    m = np.clip(m, 1.0e-15, 1.0 - 1.0e-13)

    K = ellipk(m)
    E = ellipe(m)

    # dK/dR = [E/(2(1-m)) - K/2] * d(ln m)/dR
    dlnm_dR = 1.0 / R - 2.0 * (a + R) / d2
    dK_dR = (E / (2.0 * (1.0 - m)) - 0.5 * K) * dlnm_dR

    d_K_over_D_dR = dK_dR / d - K * (a + R) / d**3

    # g_R = -dPhi/dR = (2 G M/pi) d(K/D)/dR
    return (2.0 * G / np.pi) * d_K_over_D_dR


def pringle_disk_radial_acceleration_m_s2(
    r_eval_au: np.ndarray,
    n0_cm3: float = 1.0e7,
    r0_au: float = 3600.0,
    density_power: float = -2.25,
    rstar_rsun: float = 10.0,
    height_fraction: float = 0.01,
    flaring_power: float = 1.25,
    rmax_au: float = 3600.0,
    nradial: int = 700,
    nz: int = 12,
) -> np.ndarray:
    """Aceleración radial exacta del disco axisimétrico en su plano medio.

    Modelo de densidad:

        rho(R,z) = rho_mid(R) exp[-z^2/(2 H(R)^2)]
        rho_mid(R) propto R^q
        H(R) = 0.01 R_* (R/R_*)^beta        (defaults)

    Se integra el potencial gravitatorio de anillos con espesor vertical
    Gaussiano. La integral azimutal se hace analíticamente mediante
    integrales elípticas completas; la integral vertical usa cuadratura
    Gauss-Hermite y la radial se integra numéricamente.

    A diferencia de una envolvente esférica, NO se usa GM(<R)/R: las capas
    exteriores de un disco sí ejercen una fuerza radial neta.
    """
    r_eval = np.asarray(r_eval_au, dtype=float)
    if np.any(r_eval <= 0):
        raise ValueError("r_eval_au debe contener sólo radios > 0.")
    if rmax_au <= 0:
        raise ValueError("--disk-rmax-au debe ser > 0.")
    if nradial < 100:
        raise ValueError("--disk-nradial debe ser >= 100.")
    if nz < 4:
        raise ValueError("--disk-nz debe ser >= 4.")
    if nz % 2 != 0:
        raise ValueError(
            "--disk-nz debe ser par para no colocar un nodo exactamente en z=0."
        )

    rstar_au = rstar_rsun * R_SUN_TO_AU
    if rmax_au <= rstar_au:
        raise ValueError(
            f"--disk-rmax-au={rmax_au:g} AU debe ser > R_*={rstar_au:.6g} AU."
        )

    # Radios fuente. Una malla logarítmica resuelve simultáneamente R_* y
    # escalas de cientos/miles de AU.
    a_au = np.geomspace(rstar_au, rmax_au, int(nradial))
    a_m = a_au * AU

    rho_mid = pringle_midplane_density_kg_m3(
        a_au,
        n0_cm3=n0_cm3,
        r0_au=r0_au,
        density_power=density_power,
    )
    h_m = pringle_scale_height_au(
        a_au,
        rstar_rsun=rstar_rsun,
        height_fraction=height_fraction,
        flaring_power=flaring_power,
    ) * AU

    # Gauss-Hermite:
    # integral rho_mid exp[-z^2/(2H^2)] f(z) dz
    # = rho_mid sqrt(2) H sum_i w_i f(sqrt(2) H x_i)
    xh, wh = np.polynomial.hermite.hermgauss(int(nz))
    z_m = np.sqrt(2.0) * h_m[:, None] * xh[None, :]

    # Masa por unidad de radio fuente y por nodo vertical [kg/m].
    # dM = 2*pi*a da * rho dz
    dm_da = (
        2.0
        * np.pi
        * a_m[:, None]
        * rho_mid[:, None]
        * np.sqrt(2.0)
        * h_m[:, None]
        * wh[None, :]
    )

    g = np.empty_like(r_eval, dtype=float)

    # Un loop sólo sobre radios de evaluación; cada evaluación integra todo
    # el disco de forma vectorizada en (radio fuente, z).
    for j, R_au in enumerate(r_eval):
        R_m = float(R_au * AU)
        g_per_kg = _ring_radial_accel_per_kg(
            R_m,
            a_m[:, None],
            z_m,
        )

        # Sumar cuadratura vertical y luego integrar sobre el radio fuente.
        integrand_a = np.sum(g_per_kg * dm_da, axis=1)  # m/s^2 por m de da
        g[j] = np.trapz(integrand_a, a_m)

    return g


def rotation_velocity_reference_plus_components_kms(
    r_au: np.ndarray,
    point_mass_msun: float,
    incl_deg: float,
    envelope_mass_msun_grid: np.ndarray | None = None,
    disk_g_m_s2: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocidad LOS para masa puntual + componentes gravitatorias.

    Devuelve (v_los, v2_circular), donde v2_circular está en (m/s)^2.
    La contribución del disco se suma como R*dPhi/dR = -R*g_R y puede ser
    negativa localmente si la masa exterior tira hacia afuera.
    """
    r = np.asarray(r_au, dtype=float)
    if np.any(r <= 0):
        raise ValueError("Los radios deben ser > 0 AU.")
    if point_mass_msun < 0:
        raise ValueError("La masa puntual debe ser >= 0.")

    r_m = r * AU
    v2 = G * point_mass_msun * M_SUN / r_m

    if envelope_mass_msun_grid is not None:
        menv = np.asarray(envelope_mass_msun_grid, dtype=float)
        if menv.shape != r.shape:
            raise ValueError("envelope_mass_msun_grid no coincide con r_au.")
        v2 = v2 + G * menv * M_SUN / r_m

    if disk_g_m_s2 is not None:
        gdisk = np.asarray(disk_g_m_s2, dtype=float)
        if gdisk.shape != r.shape:
            raise ValueError("disk_g_m_s2 no coincide con r_au.")
        v2 = v2 - r_m * gdisk

    sini = np.sin(np.deg2rad(incl_deg))
    if sini <= 0:
        raise ValueError("--incl debe ser > 0 grados.")

    # Si un disco extremadamente masivo produce aceleración neta hacia afuera
    # en algún punto, allí no existe una órbita circular de este modelo.
    v = np.full_like(v2, np.nan, dtype=float)
    valid = v2 > 0
    v[valid] = np.sqrt(v2[valid]) / 1000.0 * sini
    return v, v2

# -----------------------------------------------------------------------------
# Análisis principal
# -----------------------------------------------------------------------------
def analyze_pv(
    fits_file: str,
    sigma: float,
    distance: float,
    distance_unit: str,
    vsys_kms: float = 0.0,
    incl_deg: float = 90.0,
    nsigma: float = 5.0,
    center_arcsec: float = 0.0,
    discard_ranges: list[tuple[int, int]] | None = None,
    rest_freq_ghz: float | None = None,
    reference_mass_msun: float | None = None,
    plot_envelope: bool = True,
    envelope_n0_cm3: float = 1.0e7,
    envelope_r0_au: float = 3600.0,
    envelope_power: float = 1.5,
    envelope_rout_au: float | None = None,
    plot_disk: bool = True,
    disk_n0_cm3: float = 1.0e7,
    disk_r0_au: float = 3600.0,
    disk_density_power: float = -2.25,
    disk_rstar_rsun: float = 10.0,
    disk_height_fraction: float = 0.01,
    disk_flaring_power: float = 1.25,
    disk_rmax_au: float = 3600.0,
    disk_nradial: int = 700,
    disk_nz: int = 12,
    resolution_arcsec: float = 0.022,
    min_radius_au: float | None = None,
    output: str | None = "kepler_fit.png",
    show: bool = True,
):
    discard_ranges = discard_ranges or []

    # Leer FITS
    with fits.open(fits_file) as hdul:
        header = hdul[0].header.copy()
        data = np.asarray(hdul[0].data, dtype=float)

    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(
            f"Después de squeeze(), shape={data.shape}; esperaba un PV 2D."
        )

    nvel, nspace = data.shape

    # Ejes físicos
    spatial_arcsec = spatial_axis_arcsec(header, nspace)
    velocity_abs_kms = velocity_axis_kms(
        header,
        nvel,
        rest_freq_ghz=rest_freq_ghz,
    )

    # Centro espacial
    center_col = int(np.argmin(np.abs(spatial_arcsec - center_arcsec)))
    offset_arcsec = spatial_arcsec - center_arcsec

    # ------------------------------------------------------------------
    # Velocidad sistémica
    # ------------------------------------------------------------------
    # En este script:
    #   --vsys 0  => el PV está centrado espectralmente; inferir v_sys del
    #                centro geométrico del eje espectral.
    #   --vsys != 0 => usar ese valor absoluto [km/s].
    center_velocity_kms = spectral_center_velocity(velocity_abs_kms)

    if np.isclose(vsys_kms, 0.0):
        vsys_used_kms = center_velocity_kms
        vsys_mode = "automática, usando el centro espectral"
    else:
        vsys_used_kms = float(vsys_kms)
        vsys_mode = "manual"

    velocity_rel_axis_kms = velocity_abs_kms - vsys_used_kms

    # Extraer envolvente observacional
    threshold = nsigma * sigma
    rows, cols, sides = extract_envelope(
        data,
        threshold,
        center_col,
        velocity_rel_axis_kms,
    )

    if cols.size == 0:
        raise RuntimeError(
            f"No encontré píxeles válidos >= {nsigma:g} sigma = {threshold:g}."
        )

    # Offset angular -> AU
    distance_au = distance_to_au(distance, distance_unit)
    r_signed_au = offset_arcsec[cols] * distance_au / PC_TO_AU
    r_abs_au = np.abs(r_signed_au)

    vel_abs_pts = velocity_abs_kms[rows]
    vel_rel_pts = velocity_rel_axis_kms[rows]

    # Descartar columnas solicitadas
    excluded = columns_in_ranges(cols, discard_ranges)
    fit_mask = (
        (~excluded)
        & np.isfinite(r_abs_au)
        & np.isfinite(vel_rel_pts)
        & (r_abs_au > 0.0)
    )

    if np.count_nonzero(fit_mask) < 2:
        raise RuntimeError("Quedaron menos de 2 puntos para el ajuste.")

    # Ajuste Kepleriano puro a los puntos
    mass, mass_err, residuals = fit_keplerian_mass(
        r_abs_au[fit_mask],
        vel_rel_pts[fit_mask],
        incl_deg,
    )

    # Signos reales de cada rama para graficar correctamente aunque CDELT1 < 0
    def median_sign(values, mask, default):
        if np.any(mask):
            med = np.nanmedian(values[mask])
            if np.isfinite(med) and med != 0:
                return float(np.sign(med))
        return default

    left_mask = (sides == -1) & fit_mask
    right_mask = (sides == +1) & fit_mask

    left_vsign = median_sign(vel_rel_pts, left_mask, -1.0)
    right_vsign = median_sign(vel_rel_pts, right_mask, +1.0)
    left_rsign = median_sign(r_signed_au, sides == -1, -1.0)
    right_rsign = median_sign(r_signed_au, sides == +1, +1.0)

    # ------------------------------------------------------------------
    # Radio mínimo de graficado
    # ------------------------------------------------------------------
    # La integral de la envolvente se hace analíticamente desde r=0.
    # La resolución sólo limita hasta dónde dibujamos/interpretamos la curva.
    if resolution_arcsec <= 0:
        raise ValueError("--resolution-arcsec debe ser > 0.")

    resolution_au = resolution_arcsec * distance_au / PC_TO_AU
    rmin_plot_au = resolution_au if min_radius_au is None else float(min_radius_au)

    if rmin_plot_au <= 0:
        raise ValueError("--min-radius-au debe ser > 0.")

    valid_r = r_abs_au[np.isfinite(r_abs_au) & (r_abs_au > 0)]
    rmax_plot_au = float(np.nanmax(valid_r))
    rgrid_min = max(float(np.nanmin(valid_r)), rmin_plot_au)

    if rgrid_min >= rmax_plot_au:
        raise RuntimeError(
            f"El radio mínimo de graficado ({rgrid_min:.2f} AU) es >= al "
            f"radio máximo de los puntos ({rmax_plot_au:.2f} AU)."
        )

    rgrid = np.geomspace(rgrid_min, rmax_plot_au, 500)

    # Kepleriana ajustada
    sini = np.sin(np.deg2rad(incl_deg))
    vkep = V_1AU_1MSUN * np.sqrt(mass / rgrid) * sini

    # Kepleriana adicional elegida por el usuario
    if reference_mass_msun is not None:
        if reference_mass_msun <= 0:
            raise ValueError("--reference-mass debe ser > 0 M_sun.")
        vkep_reference = (
            V_1AU_1MSUN * np.sqrt(reference_mass_msun / rgrid) * sini
        )
    else:
        vkep_reference = None

    # Potencial combinado: SIEMPRE usa la masa de referencia elegida
    # por el usuario, no la masa obtenida del ajuste Kepleriano.
    vcomb_reference = None
    menv_grid = None

    if plot_envelope and reference_mass_msun is not None:
        vcomb_reference, menv_grid = rotation_velocity_point_plus_envelope_kms(
            rgrid,
            point_mass_msun=reference_mass_msun,
            incl_deg=incl_deg,
            n0_cm3=envelope_n0_cm3,
            r0_au=envelope_r0_au,
            density_power=envelope_power,
            rout_au=envelope_rout_au,
        )

    # ------------------------------------------------------------------
    # Potencial del disco de Pringle
    # ------------------------------------------------------------------
    # El disco NO se trata como una distribución esférica. Calculamos su
    # aceleración radial axisimétrica en el plano medio integrando anillos.
    gdisk_grid = None
    vref_plus_disk = None
    vref_plus_env_plus_disk = None
    disk_mass_total_msun = None

    if plot_disk:
        disk_mass_total_msun = pringle_disk_mass_msun(
            rmax_au=disk_rmax_au,
            n0_cm3=disk_n0_cm3,
            r0_au=disk_r0_au,
            density_power=disk_density_power,
            rstar_rsun=disk_rstar_rsun,
            height_fraction=disk_height_fraction,
            flaring_power=disk_flaring_power,
        )

        if reference_mass_msun is not None:
            print("Calculando potencial axisimétrico del disco de Pringle...")
            gdisk_grid = pringle_disk_radial_acceleration_m_s2(
                rgrid,
                n0_cm3=disk_n0_cm3,
                r0_au=disk_r0_au,
                density_power=disk_density_power,
                rstar_rsun=disk_rstar_rsun,
                height_fraction=disk_height_fraction,
                flaring_power=disk_flaring_power,
                rmax_au=disk_rmax_au,
                nradial=disk_nradial,
                nz=disk_nz,
            )

            # Masa de referencia + disco
            vref_plus_disk, _ = rotation_velocity_reference_plus_components_kms(
                rgrid,
                point_mass_msun=reference_mass_msun,
                incl_deg=incl_deg,
                disk_g_m_s2=gdisk_grid,
            )

            # Masa de referencia + envolvente + disco, si la envolvente está activa
            if plot_envelope:
                if menv_grid is None:
                    menv_grid = envelope_mass_msun(
                        rgrid,
                        n0_cm3=envelope_n0_cm3,
                        r0_au=envelope_r0_au,
                        density_power=envelope_power,
                        rout_au=envelope_rout_au,
                    )
                vref_plus_env_plus_disk, _ = (
                    rotation_velocity_reference_plus_components_kms(
                        rgrid,
                        point_mass_msun=reference_mass_msun,
                        incl_deg=incl_deg,
                        envelope_mass_msun_grid=menv_grid,
                        disk_g_m_s2=gdisk_grid,
                    )
                )

    # ------------------------------------------------------------------
    # Gráfica
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Colores fijos y distintos para cada conjunto/curva.
    # Cada perfil usa exactamente el mismo color en ambos cuadrantes.
    observed_color = "tab:blue"
    excluded_color = "tab:gray"
    fit_color = "tab:orange"
    reference_color = "tab:green"
    envelope_color = "tab:red"
    disk_color = "tab:purple"
    total_color = "tab:brown"

    ax.scatter(
        r_signed_au[~excluded],
        vel_rel_pts[~excluded],
        s=28,
        color=observed_color,
        label=rf"Envolvente observada $\geq {nsigma:g}\sigma$",
    )

    if np.any(excluded):
        ax.scatter(
            r_signed_au[excluded],
            vel_rel_pts[excluded],
            s=48,
            facecolors="none",
            edgecolors=excluded_color,
            linewidths=1.2,
            label="Excluidos del ajuste",
        )

    mass_label = rf"$M_\star={mass:.2f}\,M_\odot$"
    if np.isfinite(mass_err):
        mass_label += rf" $\pm$ {mass_err:.2f}"

    # Kepler pura ajustada: ambas ramas con el mismo color
    fit_line, = ax.plot(
        left_rsign * rgrid,
        left_vsign * vkep,
        lw=2,
        color=fit_color,
        label=(
            "Kepler ajustada: "
            + mass_label
            + rf", $i={incl_deg:g}^\circ$"
        ),
    )
    ax.plot(
        right_rsign * rgrid,
        right_vsign * vkep,
        lw=2,
        color=fit_color,
    )

    # Kepleriana de masa fijada por el usuario
    if vkep_reference is not None:
        ref_line, = ax.plot(
            left_rsign * rgrid,
            left_vsign * vkep_reference,
            lw=2,
            ls="--",
            color=reference_color,
            label=(
                rf"Kepler referencia: $M_\star={reference_mass_msun:g}\,M_\odot$, "
                rf"$i={incl_deg:g}^\circ$"
            ),
        )
        ax.plot(
            right_rsign * rgrid,
            right_vsign * vkep_reference,
            lw=2,
            ls="--",
            color=reference_color,
        )

        # Masa de referencia + envolvente. Esta es la única curva
        # de potencial combinado que se dibuja.
        if vcomb_reference is not None:
            ax.plot(
                left_rsign * rgrid,
                left_vsign * vcomb_reference,
                lw=2,
                ls=":",
                color=envelope_color,
                label=(
                    rf"Referencia+env.: $M_\star={reference_mass_msun:g}\,M_\odot$, "
                    rf"$n_0={envelope_n0_cm3:.1e}\,\mathrm{{cm}}^{{-3}}$, "
                    rf"$i={incl_deg:g}^\circ$"
                ),
            )
            ax.plot(
                right_rsign * rgrid,
                right_vsign * vcomb_reference,
                lw=2,
                ls=":",
                color=envelope_color,
            )

        # Masa de referencia + disco de Pringle
        if vref_plus_disk is not None:
            ax.plot(
                left_rsign * rgrid,
                left_vsign * vref_plus_disk,
                lw=2,
                ls="-.",
                color=disk_color,
                label=(
                    rf"Referencia+disco: $q={disk_density_power:g}$, "
                    rf"$R_{{\max}}={disk_rmax_au:g}$ AU, "
                    rf"$i={incl_deg:g}^\circ$"
                ),
            )
            ax.plot(
                right_rsign * rgrid,
                right_vsign * vref_plus_disk,
                lw=2,
                ls="-.",
                color=disk_color,
            )

        # Masa de referencia + envolvente + disco
        if vref_plus_env_plus_disk is not None:
            total_ls = (0, (5, 1, 1, 1, 1, 1))
            ax.plot(
                left_rsign * rgrid,
                left_vsign * vref_plus_env_plus_disk,
                lw=2.2,
                ls=total_ls,
                color=total_color,
                label=(
                    rf"Referencia+env.+disco: $M_\star={reference_mass_msun:g}\,M_\odot$, "
                    rf"$i={incl_deg:g}^\circ$"
                ),
            )
            ax.plot(
                right_rsign * rgrid,
                right_vsign * vref_plus_env_plus_disk,
                lw=2.2,
                ls=total_ls,
                color=total_color,
            )

    ax.axhline(
        0.0, ls="--", lw=1.2, color="black", alpha=0.7,
        label=r"$v-v_{\rm sys}=0$"
    )
    ax.axvline(0.0, ls=":", lw=1.2, color="black", alpha=0.7)

    ax.set_xlabel("Offset espacial proyectado [AU]")
    ax.set_ylabel(r"$v-v_{\rm sys}$ [km/s]")
    ax.set_title(Path(fits_file).name)
    ax.grid(alpha=0.25)

    # Eje y fijo a 1.5 veces el rango total del PV
    vmin_pv = np.nanmin(velocity_rel_axis_kms)
    vmax_pv = np.nanmax(velocity_rel_axis_kms)
    vcenter_pv = 0.5 * (vmin_pv + vmax_pv)
    vrange_pv = vmax_pv - vmin_pv
    if np.isfinite(vrange_pv) and vrange_pv > 0:
        ax.set_ylim(
            vcenter_pv - 0.75 * vrange_pv,
            vcenter_pv + 0.75 * vrange_pv,
        )

    ax.legend(fontsize=9)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Figura guardada en: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # ------------------------------------------------------------------
    # Resumen
    # ------------------------------------------------------------------
    print("\n------------- RESULTADOS -------------")
    print(f"FITS: {fits_file}")
    print(f"Shape (vel, espacio): {data.shape}")
    print(f"Columna central: {center_col} (0-based)")
    print(f"Umbral: {threshold:g} = {nsigma:g} x sigma")
    print(f"Puntos extraídos: {len(cols)}")
    print(f"Puntos descartados: {np.count_nonzero(excluded)}")
    print(f"Puntos usados: {np.count_nonzero(fit_mask)}")
    print()
    print(f"Velocidad del centro espectral: {center_velocity_kms:.5f} km/s")
    print(f"v_sys usada: {vsys_used_kms:.5f} km/s ({vsys_mode})")
    print()
    print(f"Masa Kepleriana ajustada: {mass:.4g} M_sun")
    if np.isfinite(mass_err):
        print(f"Incertidumbre aproximada: +/- {mass_err:.3g} M_sun")
    print(f"Inclinación: {incl_deg:g} deg")

    if reference_mass_msun is not None:
        print(f"Masa Kepleriana de referencia: {reference_mass_msun:g} M_sun")

    print()
    print(f"Radio mínimo dibujado: {rgrid_min:.3f} AU")
    print(
        f"Resolución adoptada: {resolution_arcsec:g} arcsec = "
        f"{resolution_au:.3f} AU"
    )

    if plot_envelope:
        if reference_mass_msun is None:
            print()
            print(
                "NOTA: --no-envelope no está activo, pero no se proporcionó "
                "--reference-mass. No se dibujó la curva masa puntual + envolvente."
            )

        menv_r0 = float(envelope_mass_msun(
            envelope_r0_au,
            n0_cm3=envelope_n0_cm3,
            r0_au=envelope_r0_au,
            density_power=envelope_power,
            rout_au=envelope_rout_au,
        ))
        menv_rmax = float(envelope_mass_msun(
            rmax_plot_au,
            n0_cm3=envelope_n0_cm3,
            r0_au=envelope_r0_au,
            density_power=envelope_power,
            rout_au=envelope_rout_au,
        ))

        print()
        print("Envolvente esférica:")
        print(
            f"  n(r) = {envelope_n0_cm3:.3g} "
            f"(r/{envelope_r0_au:g} AU)^(-{envelope_power:g}) cm^-3"
        )
        print("  masa por partícula = 2 m_H (H2)")
        print(f"  M_env(<{envelope_r0_au:g} AU) = {menv_r0:.4g} M_sun")
        print(f"  M_env(<{rmax_plot_au:.3f} AU) = {menv_rmax:.4g} M_sun")
        if envelope_rout_au is not None:
            print(f"  R_out = {envelope_rout_au:g} AU")
        else:
            print("  R_out no fijado; el perfil se usa hasta el máximo radio graficado")

    if plot_disk:
        print()
        print("Disco de Pringle / flared:")
        print(
            f"  n_mid(R) = {disk_n0_cm3:.3g} "
            f"(R/{disk_r0_au:g} AU)^({disk_density_power:g}) cm^-3"
        )
        print(
            f"  R_* = {disk_rstar_rsun:g} R_sun = "
            f"{disk_rstar_rsun * R_SUN_TO_AU:.6g} AU"
        )
        print(
            f"  H(R_*) = {disk_height_fraction:g} R_*; "
            f"H(R) propto R^{disk_flaring_power:g}"
        )
        print(f"  R_max = {disk_rmax_au:g} AU")
        if disk_mass_total_msun is not None:
            print(f"  M_disk(R_*..R_max) = {disk_mass_total_msun:.4g} M_sun")
        if reference_mass_msun is None:
            print(
                "  NOTA: no se proporcionó --reference-mass; se calculó la "
                "masa del disco, pero no se dibujó una curva estrella+disco."
            )

    print("--------------------------------------\n")

    return {
        "rows": rows,
        "cols": cols,
        "sides": sides,
        "r_signed_au": r_signed_au,
        "r_abs_au": r_abs_au,
        "velocity_abs_kms": vel_abs_pts,
        "velocity_rel_kms": vel_rel_pts,
        "excluded": excluded,
        "fit_mask": fit_mask,
        "mass_msun": mass,
        "mass_err_msun": mass_err,
        "center_col": center_col,
        "center_velocity_kms": center_velocity_kms,
        "vsys_used_kms": vsys_used_kms,
        "threshold": threshold,
        "residuals": residuals,
        "reference_mass_msun": reference_mass_msun,
        "resolution_au": resolution_au,
        "rgrid_au": rgrid,
        "envelope_mass_grid_msun": menv_grid,
        "combined_velocity_reference_kms": vcomb_reference,
        "disk_radial_acceleration_m_s2": gdisk_grid,
        "disk_mass_total_msun": disk_mass_total_msun,
        "reference_plus_disk_velocity_kms": vref_plus_disk,
        "reference_plus_envelope_plus_disk_velocity_kms": vref_plus_env_plus_disk,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extrae una envolvente de 5 sigma de un PV, ajusta una Kepleriana "
            "y puede graficar potenciales combinados de una masa puntual, "
            "una envolvente esférica y un disco de Pringle/flared axisimétrico."
        )
    )

    parser.add_argument(
        "fits_file",
        nargs="?",
        default="../pv/pv_G328_resampled_flipped.fits",
        help="Archivo FITS del PV",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=2.3e-3,
        help="RMS/sigma; default=2.3e-3",
    )

    parser.add_argument(
        "--nsigma",
        type=float,
        default=5.0,
        help="Multiplicador del RMS; default=5",
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=2500.0,
        help="Distancia a la fuente; default=2500",
    )

    parser.add_argument(
        "--distance-unit",
        choices=["pc", "kpc", "au"],
        default="pc",
        help="Unidad de --distance; default=pc",
    )

    parser.add_argument(
        "--vsys",
        type=float,
        default=0.0,
        help=(
            "v_sys absoluta [km/s]. Si vale 0 (default), se asume que el PV "
            "está centrado y se obtiene v_sys del centro espectral."
        ),
    )

    parser.add_argument(
        "--incl",
        type=float,
        default=90.0,
        help="Inclinación: 0=face-on, 90=edge-on; default=90",
    )

    parser.add_argument(
        "--center-arcsec",
        type=float,
        default=0.0,
        help="Centro espacial [arcsec]; default=0",
    )

    parser.add_argument(
        "--discard",
        nargs="*",
        default=[],
        metavar="COL0:COL1",
        help="Columnas a excluir. Ej.: --discard 55:70 80:82",
    )

    parser.add_argument(
        "--rest-freq-ghz",
        type=float,
        default=None,
        help="REST frequency [GHz] si el header no tiene RESTFRQ",
    )

    parser.add_argument(
        "--reference-mass",
        "--plot-mass",
        dest="reference_mass",
        type=float,
        default=None,
        help=(
            "Masa [M_sun] de una Kepleriana adicional a graficar. "
            "También se dibuja esa misma masa puntual + envolvente."
        ),
    )

    # Parámetros de la envolvente esférica
    parser.add_argument(
        "--no-envelope",
        action="store_true",
        help="No dibujar las curvas de masa puntual + envolvente.",
    )

    parser.add_argument(
        "--envelope-n0",
        type=float,
        default=1.0e7,
        help=(
            "Densidad numérica H2 n0 [cm^-3] en r0; default=1e7."
        ),
    )

    parser.add_argument(
        "--envelope-r0-au",
        type=float,
        default=12357,
        help="Radio de normalización de la densidad [AU]; default=3600",
    )

    parser.add_argument(
        "--envelope-power",
        type=float,
        default=1.5,
        help="Exponente p en n(r) propto r^-p; default=1.5",
    )

    parser.add_argument(
        "--envelope-rout-au",
        type=float,
        default=None,
        help=(
            "Radio externo de la envolvente [AU]. Si se omite, el perfil "
            "continúa hasta el mayor radio graficado."
        ),
    )

    # Parámetros del disco de Pringle / flared
    parser.add_argument(
        "--no-disk",
        action="store_true",
        help="No calcular ni dibujar el potencial del disco de Pringle.",
    )

    parser.add_argument(
        "--disk-n0",
        type=float,
        default=1.0e7,
        help=(
            "Densidad numérica H2 en el plano medio [cm^-3] en disk-r0-au; "
            "default=1e7."
        ),
    )

    parser.add_argument(
        "--disk-r0-au",
        type=float,
        default=3600.0,
        help="Radio de normalización de n_mid [AU]; default=3600.",
    )

    parser.add_argument(
        "--disk-power",
        type=float,
        default=-2.25,
        help="Exponente q en n_mid(R) propto R^q; default=-2.25.",
    )

    parser.add_argument(
        "--disk-rstar-rsun",
        type=float,
        default=10.0,
        help="Radio estelar R_* [R_sun]; default=10.",
    )

    parser.add_argument(
        "--disk-height-fraction",
        type=float,
        default=0.01,
        help="H(R_*)/R_*; default=0.01.",
    )

    parser.add_argument(
        "--disk-flaring-power",
        type=float,
        default=1.25,
        help=(
            "beta en H(R)=H(R_*)(R/R_*)^beta; default=1.25. "
            "Con q=-2.25 corresponde a la pareja alpha=beta+1."
        ),
    )

    parser.add_argument(
        "--disk-rmax-au",
        type=float,
        default=3600.0,
        help="Radio externo del disco [AU]; default=3600.",
    )

    parser.add_argument(
        "--disk-nradial",
        type=int,
        default=700,
        help="Número de radios fuente para integrar el disco; default=700.",
    )

    parser.add_argument(
        "--disk-nz",
        type=int,
        default=12,
        help=(
            "Orden par de la cuadratura Gauss-Hermite vertical; default=12."
        ),
    )

    # Resolución: se usa sólo como radio mínimo de GRAFICADO, no de integración
    parser.add_argument(
        "--resolution-arcsec",
        type=float,
        default=0.022,
        help=(
            "Resolución angular usada como radio mínimo de graficado; "
            "default=0.022 arcsec. A 2500 pc corresponde a 55 AU."
        ),
    )

    parser.add_argument(
        "--min-radius-au",
        type=float,
        default=None,
        help=(
            "Radio mínimo de graficado [AU]. Si se da, reemplaza al calculado "
            "con --resolution-arcsec. No modifica la integral desde r=0."
        ),
    )

    parser.add_argument(
        "--output",
        default="kepler_fit.png",
        help="Figura de salida; default=kepler_fit.png",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No mostrar la figura en pantalla",
    )

    args = parser.parse_args()

    analyze_pv(
        fits_file=args.fits_file,
        sigma=args.sigma,
        nsigma=args.nsigma,
        distance=args.distance,
        distance_unit=args.distance_unit,
        vsys_kms=args.vsys,
        incl_deg=args.incl,
        center_arcsec=args.center_arcsec,
        discard_ranges=parse_ranges(args.discard),
        rest_freq_ghz=args.rest_freq_ghz,
        reference_mass_msun=args.reference_mass,
        plot_envelope=not args.no_envelope,
        envelope_n0_cm3=args.envelope_n0,
        envelope_r0_au=args.envelope_r0_au,
        envelope_power=args.envelope_power,
        envelope_rout_au=args.envelope_rout_au,
        plot_disk=not args.no_disk,
        disk_n0_cm3=args.disk_n0,
        disk_r0_au=args.disk_r0_au,
        disk_density_power=args.disk_power,
        disk_rstar_rsun=args.disk_rstar_rsun,
        disk_height_fraction=args.disk_height_fraction,
        disk_flaring_power=args.disk_flaring_power,
        disk_rmax_au=args.disk_rmax_au,
        disk_nradial=args.disk_nradial,
        disk_nz=args.disk_nz,
        resolution_arcsec=args.resolution_arcsec,
        min_radius_au=args.min_radius_au,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
