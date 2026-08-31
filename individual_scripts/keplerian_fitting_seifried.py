#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------
G = 6.67430e-11               # m^3 kg^-1 s^-2
M_SUN = 1.98847e30            # kg
M_H = 1.6735575e-27           # kg
M_H2 = 2.0 * M_H              # kg; cada partícula de la envolvente es H2
AU = 1.495978707e11           # m
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
    # Gráfica
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.scatter(
        r_signed_au[~excluded],
        vel_rel_pts[~excluded],
        s=28,
        label=rf"Envolvente observada $\geq {nsigma:g}\sigma$",
    )

    if np.any(excluded):
        ax.scatter(
            r_signed_au[excluded],
            vel_rel_pts[excluded],
            s=48,
            facecolors="none",
            edgecolors="black",
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
        color=fit_line.get_color(),
    )

    # Kepleriana de masa fijada por el usuario
    if vkep_reference is not None:
        ref_line, = ax.plot(
            left_rsign * rgrid,
            left_vsign * vkep_reference,
            lw=2,
            ls="--",
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
            color=ref_line.get_color(),
        )

        # Masa de referencia + envolvente. Esta es la única curva
        # de potencial combinado que se dibuja.
        if vcomb_reference is not None:
            ax.plot(
                left_rsign * rgrid,
                left_vsign * vcomb_reference,
                lw=2,
                ls=":",
                color=ref_line.get_color(),
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
                color=ref_line.get_color(),
            )

    ax.axhline(0.0, ls="--", lw=1.2, label=r"$v-v_{\rm sys}=0$")
    ax.axvline(0.0, ls=":", lw=1.2)

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

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),
        bbox_transform=ax.transAxes,
        fontsize=8,
        labelspacing=0.4,
        handlelength=1.5,
        frameon=True
    )
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
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extrae una envolvente de 5 sigma de un PV, ajusta una Kepleriana "
            "y puede graficar el potencial combinado de una masa puntual + "
            "una envolvente esférica n(r) proporcional a r^-1.5."
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
        "--ref-mass",
        "--plot-mass",
        dest="reference_mass",
        type=float,
        default=20,
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
        default=12376,
        help="Radio de normalización de la densidad [AU]; default=3600",
    )

    parser.add_argument(
        "--env-pow",
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
        envelope_power=args.env_pow,
        envelope_rout_au=args.envelope_rout_au,
        resolution_arcsec=args.resolution_arcsec,
        min_radius_au=args.min_radius_au,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()