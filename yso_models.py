'''
Script that creates possible models of molecular emission for RADMC-3D
'''

#------------------
#Import the package
#------------------

from sf3dmodels import Model, Plot_model
from sf3dmodels import Resolution as Res
import sf3dmodels.utils.units as u
import astropy.units as U
import sf3dmodels.rt as rt        
import sf3dmodels.utils.constants as ct            
from astropy.constants import G
#-----------------
#Extra libraries
#-----------------
from matplotlib import colors
import numpy as np
import os
import time


    #********************************
    #Write molecule number density file
    #********************************

def write_molecule_files(nx, ny, nz, density, prop, molec=''):
    with open('numberdens_%s.inp' % molec, 'w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n' % (nx * ny * nz))     # Nr of cells
        data = prop['abundance'] * density.total * (100**-3)  # To 1/cm3 units
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')

    # Lines file
    with open('lines.inp', 'w+') as f:
        f.write('2\n')
        f.write('1\n')
        f.write('%s    leiden    0    0    0\n' % molec)

    # Dust opacity control file
    with open('dustopac.inp', 'w+') as f:
        f.write('2               Format number of this file\n')
        f.write('1               Nr of dust species\n')
        f.write('============================================================================\n')
        f.write('1               Way in which this dust species is read\n')
        f.write('0               0=Thermal grain\n')
        f.write('silicate        Extension of name of dustkappa_***.inp file\n')
        f.write('----------------------------------------------------------------------------\n')


import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic


def _safe_lognorm(values, floor=1e-99):
    """
    Construye una normalización logarítmica evitando errores si hay ceros,
    negativos, NaNs o rangos degenerados.
    """
    values = np.asarray(values)
    good = np.isfinite(values) & (values > 0)

    if not np.any(good):
        return None

    vmin = np.nanmin(values[good])
    vmax = np.nanmax(values[good])

    vmin = max(vmin, floor)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = max(vmax / 1e3, floor)

    return LogNorm(vmin=vmin, vmax=vmax)


def _binned_profile(r_values, prop_values, r_bins, statistic="median"):
    """
    Calcula un perfil radial evitando NaNs e infinitos.
    """
    r_values = np.asarray(r_values, dtype=float)
    prop_values = np.asarray(prop_values, dtype=float)

    valid = np.isfinite(r_values) & np.isfinite(prop_values)

    prof, _, _ = binned_statistic(
        r_values[valid],
        prop_values[valid],
        statistic=statistic,
        bins=r_bins,
    )

    return prof


def plot_ulrichdisk_diagnostics(
    GRID,
    prop,
    density,
    MStar_msun,
    Rdisc_au=None,
    Renv_au=None,
    tag="",
    output_dir=".",
    n_random=6000,
    show=False,
    seed=1234,
):
    """
    Genera plots de diagnóstico para el modelo Ulrich + disco.

    Parámetros
    ----------
    GRID : objeto GRID de sf3dmodels
        Debe contener GRID.XYZ, GRID.NPoints y GRID.step.

    prop : dict
        Diccionario del modelo. Se espera:
            prop["temp_dust"]
            prop["velocity"] = [vx, vy, vz]

    density : objeto density de sf3dmodels
        Se espera density.total.

    MStar_msun : float
        Masa del objeto central en masas solares.

    Rdisc_au : float, opcional
        Radio del disco en AU.

    Renv_au : float, opcional
        Radio externo de la envolvente en AU.

    tag : str
        Sufijo para los nombres de salida.

    output_dir : str
        Carpeta donde se guardan los plots.

    n_random : int
        Número máximo de puntos para el scatter 3D.

    show : bool
        Si True, muestra las figuras en pantalla.

    seed : int
        Semilla para reproducibilidad del muestreo 3D.
    """

    print("Generando plots de diagnóstico...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{tag}" if tag else ""

    # ============================================================
    # Coordenadas del grid
    # ============================================================

    x_coords = np.asarray(GRID.XYZ[0], dtype=float)
    y_coords = np.asarray(GRID.XYZ[1], dtype=float)
    z_coords = np.asarray(GRID.XYZ[2], dtype=float)

    NPoints = GRID.NPoints

    # ============================================================
    # Masa total del gas
    # ============================================================

    dx, dy, dz = GRID.step
    dv = dx * dy * dz

    print("Step in AU =", (dx * U.m).to(U.au))

    dv = (dv * U.m**3).to(U.cm**3)

    total_part = np.sum(density.total * dv) / (1e6 * U.cm**3)
    total_mass = (total_part * 3.32e-24 * U.g).to(U.M_sun)

    # Las coordenadas de sf3dmodels están en metros.
    x_au_all = x_coords / u.au
    y_au_all = y_coords / u.au
    z_au_all = z_coords / u.au

    # Radio esférico.
    r_m = np.sqrt(
        x_coords**2
        + y_coords**2
        + z_coords**2
    )
    r_au = r_m / u.au

    box_half_size_au = np.nanmax(
        [
            np.nanmax(np.abs(x_au_all)),
            np.nanmax(np.abs(y_au_all)),
            np.nanmax(np.abs(z_au_all)),
        ]
    )

    if Renv_au is not None:
        rmax_plot_au = Renv_au
    else:
        rmax_plot_au = box_half_size_au

    # ============================================================
    # Propiedades físicas
    # ============================================================

    n_cm3 = np.asarray(density.total, dtype=float) / 1e6
    T_K = np.asarray(prop["temp_dust"], dtype=float)

    vx = np.asarray(prop["velocity"][0], dtype=float)
    vy = np.asarray(prop["velocity"][1], dtype=float)
    vz = np.asarray(prop["velocity"][2], dtype=float)

    v_kms = np.sqrt(
        vx**2
        + vy**2
        + vz**2
    ) / 1e3

    # ============================================================
    # Corte XY en z ≈ 0
    # ============================================================

    z_unique = np.unique(z_coords)
    z0 = z_unique[np.argmin(np.abs(z_unique))]

    if len(z_unique) > 1:
        dz_min = np.nanmin(
            np.abs(np.diff(np.sort(z_unique)))
        )
        z_tol = 0.1 * dz_min
    else:
        z_tol = 0.0

    slice_mask = np.isclose(
        z_coords,
        z0,
        rtol=0.0,
        atol=z_tol,
    )

    print(f"Puntos en el corte z≈0: {np.sum(slice_mask)}")

    if np.sum(slice_mask) > 0:
        x_slice = x_coords[slice_mask] / u.au
        y_slice = y_coords[slice_mask] / u.au

        n_slice = n_cm3[slice_mask]
        T_slice = T_K[slice_mask]

        vx_slice = vx[slice_mask]
        vy_slice = vy[slice_mask]
        vz_slice = vz[slice_mask]

        v_slice_kms = np.sqrt(
            vx_slice**2
            + vy_slice**2
            + vz_slice**2
        ) / 1e3

        # Radio cilíndrico en el plano medio.
        r_midplane_au = np.sqrt(
            x_slice**2
            + y_slice**2
        )

        x_unique = np.unique(x_slice)
        y_unique = np.unique(y_slice)

        nx2d = len(x_unique)
        ny2d = len(y_unique)

        print(
            "2D slice shape inferred from coordinates: "
            f"({ny2d}, {nx2d})"
        )

        expected_size = nx2d * ny2d

        if expected_size != len(x_slice):
            print(
                "Advertencia: el corte no parece ser una "
                "malla rectangular."
            )
            print(
                "Se usará scatter en lugar de pcolormesh "
                "para el corte 2D."
            )

            use_scatter = True

        else:
            use_scatter = False

            # Ordenar primero por y y después por x.
            sort_idx = np.lexsort(
                (x_slice, y_slice)
            )

            x2d = x_slice[sort_idx].reshape(
                ny2d,
                nx2d,
            )

            y2d = y_slice[sort_idx].reshape(
                ny2d,
                nx2d,
            )

            n2d = n_slice[sort_idx].reshape(
                ny2d,
                nx2d,
            )

            T2d = T_slice[sort_idx].reshape(
                ny2d,
                nx2d,
            )

        # Tres filas:
        # fila 0: mapas XY
        # fila 1: perfiles esféricos 3D
        # fila 2: perfiles en z ≈ 0

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(13, 15),
        )

        mass_text = (
            rf"Total gas mass: "
            rf"$M_{{\rm gas}} = "
            rf"{total_mass.to_value(U.M_sun):.3e}\ M_\odot$"
        )

        fig.suptitle(
            mass_text,
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

        # ========================================================
        # Mapas 2D en el plano XY
        # ========================================================

        # Densidad
        norm_n = _safe_lognorm(
            n_slice,
            floor=1e-12,
        )

        if use_scatter:
            im1 = axes[0, 0].scatter(
                x_slice,
                y_slice,
                c=n_slice,
                s=4,
                norm=norm_n,
                cmap="viridis",
            )

        else:
            n2d_plot = np.ma.masked_where(
                n2d <= 0,
                n2d,
            )

            im1 = axes[0, 0].pcolormesh(
                x2d,
                y2d,
                n2d_plot,
                shading="auto",
                norm=norm_n,
                cmap="viridis",
            )

        axes[0, 0].set_xlabel("X [AU]")
        axes[0, 0].set_ylabel("Y [AU]")

        axes[0, 0].set_title(
            r"Density [cm$^{-3}$] — XY slice, $z\approx0$"
        )

        axes[0, 0].set_aspect("equal")

        plt.colorbar(
            im1,
            ax=axes[0, 0],
            label=r"$n_{\rm H_2}$ [cm$^{-3}$]",
        )

        # Temperatura
        norm_T = _safe_lognorm(
            T_slice,
            floor=1.0,
        )

        if use_scatter:
            im2 = axes[0, 1].scatter(
                x_slice,
                y_slice,
                c=T_slice,
                s=4,
                norm=norm_T,
                cmap="hot",
            )

        else:
            T2d_plot = np.ma.masked_where(
                T2d <= 0,
                T2d,
            )

            im2 = axes[0, 1].pcolormesh(
                x2d,
                y2d,
                T2d_plot,
                shading="auto",
                norm=norm_T,
                cmap="hot",
            )

        axes[0, 1].set_xlabel("X [AU]")
        axes[0, 1].set_ylabel("Y [AU]")

        axes[0, 1].set_title(
            r"Temperature [K] — XY slice, $z\approx0$"
        )

        axes[0, 1].set_aspect("equal")

        plt.colorbar(
            im2,
            ax=axes[0, 1],
            label="T [K]",
        )

        # ========================================================
        # Bins radiales
        # ========================================================

        nbins = 120

        r_bins = np.linspace(
            0.0,
            rmax_plot_au,
            nbins + 1,
        )

        r_centers = 0.5 * (
            r_bins[:-1]
            + r_bins[1:]
        )

        # Velocidad kepleriana.
        v_kep_kms = (
            np.sqrt(
                G
                * (MStar_msun * U.M_sun)
                / (r_centers * U.au)
            )
            .to(U.km / U.s)
            .value
        )

        # ========================================================
        # Perfiles radiales esféricos 3D
        # ========================================================

        dens_med_sph = _binned_profile(
            r_au,
            n_cm3,
            r_bins,
            statistic="median",
        )

        temp_med_sph = _binned_profile(
            r_au,
            T_K,
            r_bins,
            statistic="median",
        )

        vel_med_sph = _binned_profile(
            r_au,
            v_kms,
            r_bins,
            statistic="median",
        )

        # Densidad esférica
        ax_sph_dens = axes[1, 0]

        ax_sph_dens.plot(
            r_centers,
            dens_med_sph,
            lw=2,
            label="spherical median",
        )

        if Rdisc_au is not None:
            ax_sph_dens.axvline(
                Rdisc_au,
                color="g",
                ls="--",
                alpha=0.8,
                label=r"$R_{\rm disc}$",
            )

        if Renv_au is not None:
            ax_sph_dens.axvline(
                Renv_au,
                color="orange",
                ls="--",
                alpha=0.8,
                label=r"$R_{\rm env}$",
            )

        ax_sph_dens.set_xlabel(
            "Spherical radius [AU]"
        )

        ax_sph_dens.set_ylabel(
            r"Density [cm$^{-3}$]"
        )

        ax_sph_dens.set_yscale("log")
        ax_sph_dens.set_xlim(0, rmax_plot_au)

        ax_sph_dens.set_title(
            "Spherical radial density profile"
        )

        ax_sph_dens.grid(
            True,
            alpha=0.3,
        )

        ax_sph_dens.legend()

        # Temperatura y velocidad esféricas
        ax_sph_tv = axes[1, 1]

        ax_sph_tv.plot(
            r_centers,
            temp_med_sph,
            lw=2,
            label="T spherical median [K]",
        )

        ax_sph_tv.plot(
            r_centers,
            vel_med_sph,
            lw=2,
            ls=":",
            label=r"v spherical median [km s$^{-1}$]",
        )

        if Rdisc_au is not None:
            ax_sph_tv.axvline(
                Rdisc_au,
                color="g",
                ls="--",
                alpha=0.8,
            )

        if Renv_au is not None:
            ax_sph_tv.axvline(
                Renv_au,
                color="orange",
                ls="--",
                alpha=0.8,
            )

        ax_sph_tv.set_xlabel(
            "Spherical radius [AU]"
        )

        ax_sph_tv.set_ylabel(
            "Temperature / Velocity"
        )

        ax_sph_tv.set_yscale("log")
        ax_sph_tv.set_xlim(0, rmax_plot_au)

        ax_sph_tv.set_title(
            "Spherical radial temperature and velocity profiles"
        )

        ax_sph_tv.grid(
            True,
            alpha=0.3,
        )

        ax_sph_tv.legend()

        # ========================================================
        # Perfiles radiales en el plano medio
        # ========================================================

        dens_med_mid = _binned_profile(
            r_midplane_au,
            n_slice,
            r_bins,
            statistic="median",
        )

        temp_med_mid = _binned_profile(
            r_midplane_au,
            T_slice,
            r_bins,
            statistic="median",
        )

        vel_med_mid = _binned_profile(
            r_midplane_au,
            v_slice_kms,
            r_bins,
            statistic="median",
        )

        # Densidad del plano medio
        ax_mid_dens = axes[2, 0]

        ax_mid_dens.plot(
            r_centers,
            dens_med_mid,
            lw=2,
            label=r"midplane median, $z\approx0$",
        )

        if Rdisc_au is not None:
            ax_mid_dens.axvline(
                Rdisc_au,
                color="g",
                ls="--",
                alpha=0.8,
                label=r"$R_{\rm disc}$",
            )

        if Renv_au is not None:
            ax_mid_dens.axvline(
                Renv_au,
                color="orange",
                ls="--",
                alpha=0.8,
                label=r"$R_{\rm env}$",
            )

        ax_mid_dens.set_xlabel(
            "Cylindrical radius in midplane [AU]"
        )

        ax_mid_dens.set_ylabel(
            r"Density [cm$^{-3}$]"
        )

        ax_mid_dens.set_yscale("log")
        ax_mid_dens.set_xlim(0, rmax_plot_au)

        ax_mid_dens.set_title(
            r"Midplane radial density profile, $z\approx0$"
        )

        ax_mid_dens.grid(
            True,
            alpha=0.3,
        )

        ax_mid_dens.legend()

        # Temperatura y velocidad en el plano medio
        ax_mid_tv = axes[2, 1]

        ax_mid_tv.plot(
            r_centers,
            temp_med_mid,
            lw=2,
            label="T midplane median [K]",
        )

        ax_mid_tv.plot(
            r_centers,
            vel_med_mid,
            lw=2,
            ls=":",
            label=r"$v_\phi$ midplane median [km s$^{-1}$]",
        )

        # La velocidad kepleriana se muestra únicamente
        # en el panel del plano medio.
        ax_mid_tv.plot(
            r_centers,
            v_kep_kms,
            color="crimson",
            lw=2,
            ls="--",
            label=(
                rf"$v_{{\rm Kep}}$ "
                rf"($M_\star={MStar_msun:g}\,M_\odot$)"
            ),
        )

        if Rdisc_au is not None:
            ax_mid_tv.axvline(
                Rdisc_au,
                color="g",
                ls="--",
                alpha=0.8,
            )

        if Renv_au is not None:
            ax_mid_tv.axvline(
                Renv_au,
                color="orange",
                ls="--",
                alpha=0.8,
            )

        ax_mid_tv.set_xlabel(
            "Cylindrical radius in midplane [AU]"
        )

        ax_mid_tv.set_ylabel(
            "Temperature [K] / Velocity [km s$^{-1}$]"
        )

        ax_mid_tv.set_yscale("log")
        ax_mid_tv.set_xlim(0, rmax_plot_au)

        ax_mid_tv.set_title(
            r"Midplane radial temperature and velocity profiles, "
            r"$z\approx0$"
        )

        ax_mid_tv.grid(
            True,
            alpha=0.3,
        )

        ax_mid_tv.legend()

        plt.tight_layout()

        diag_name = (
            output_dir
            / f"ulrichdisk_diagnostics{suffix}.png"
        )

        plt.savefig(
            diag_name,
            dpi=180,
            bbox_inches="tight",
        )

        if show:
            plt.show()
        else:
            plt.close(fig)

        print(
            f"Plot guardado como '{diag_name}'"
        )

    else:
        print(
            "No fue posible construir el corte z≈0"
        )

    # ============================================================
    # Visualización 3D
    # ============================================================

    try:
        print("Generando visualización 3D...")

        rng = np.random.default_rng(seed)

        weights = n_cm3.copy() ** 2

        weights[~np.isfinite(weights)] = 0.0
        weights[weights < 0.0] = 0.0

        positive = weights > 0.0

        if np.sum(positive) == 0:
            n_points = min(
                n_random,
                NPoints,
            )

            indices = rng.choice(
                NPoints,
                size=n_points,
                replace=False,
            )

        else:
            available = np.count_nonzero(
                positive
            )

            n_points = min(
                n_random,
                available,
            )

            candidate_indices = np.where(
                positive
            )[0]

            weights_pos = weights[positive]
            weights_pos /= np.sum(weights_pos)

            indices = rng.choice(
                candidate_indices,
                size=n_points,
                replace=False,
                p=weights_pos,
            )

        x_plot = x_coords[indices] / u.au
        y_plot = y_coords[indices] / u.au
        z_plot = z_coords[indices] / u.au

        d_plot = n_cm3[indices]
        t_plot = T_K[indices]

        fig = plt.figure(
            figsize=(15, 6)
        )

        # Densidad 3D
        ax1 = fig.add_subplot(
            121,
            projection="3d",
        )

        norm_d3d = _safe_lognorm(
            d_plot,
            floor=1e-8,
        )

        sc1 = ax1.scatter(
            x_plot,
            y_plot,
            z_plot,
            c=d_plot,
            s=4,
            alpha=0.6,
            cmap="viridis",
            norm=norm_d3d,
        )

        ax1.set_xlabel("X [AU]")
        ax1.set_ylabel("Y [AU]")
        ax1.set_zlabel("Z [AU]")

        ax1.set_title(
            r"Density [cm$^{-3}$]"
        )

        plt.colorbar(
            sc1,
            ax=ax1,
            label=r"$n_{\rm H_2}$ [cm$^{-3}$]",
        )

        # Temperatura 3D
        ax2 = fig.add_subplot(
            122,
            projection="3d",
        )

        norm_t3d = _safe_lognorm(
            t_plot,
            floor=1.0,
        )

        sc2 = ax2.scatter(
            x_plot,
            y_plot,
            z_plot,
            c=t_plot,
            s=4,
            alpha=0.6,
            cmap="hot",
            norm=norm_t3d,
        )

        ax2.set_xlabel("X [AU]")
        ax2.set_ylabel("Y [AU]")
        ax2.set_zlabel("Z [AU]")
        ax2.set_title("Temperature [K]")

        plt.colorbar(
            sc2,
            ax=ax2,
            label="T [K]",
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.97]
        )

        plot3d_name = (
            output_dir
            / f"ulrichdisk_3d{suffix}.png"
        )

        plt.savefig(
            plot3d_name,
            dpi=180,
            bbox_inches="tight",
        )

        if show:
            plt.show()
        else:
            plt.close(fig)

        print(
            "Visualización 3D guardada como "
            f"'{plot3d_name}'"
        )

    except Exception as err:
        print(
            "No fue posible generar la visualización 3D."
        )
        print(f"Error: {err}")
        
def UlrichDisk(nmodel, *, MStar, MRate, Rdisc, Arho0, Renv, cavity_ang, exp_disc, molec_abund, BT, T10Env, p, molec, grid_config, model_config, radmc_config, prop_only=False, diagnostic_plots=True, diagnostic_tag="Main", diagnostic_output_dir="."):
    t0 = time.time()
    if grid_config is None:
        raise ValueError("grid_config must be provided")

    if model_config is None:
        raise ValueError("model_config must be provided")

    if radmc_config is None:
        raise ValueError("radmc_config must be provided")
        t0 = time.time()

    discFlag = model_config["disc"]
    envFlag = model_config["envelope"]

    gtd_ratio = model_config["gas_to_dust_ratio"]
    microturbulence = model_config["microturbulence_ms"]
    rho_min_env = model_config["rho_min_env"]

    print('\n')
    print('Passed paremeters are:')
    print('nmodel: {}'.format(nmodel))
    print('MStar: {}'.format(MStar))
    print('MRate: {}'.format(MRate))
    print('Rdisc: {}'.format(Rdisc))
    print('Arho0: {}'.format(Arho0))
    print('Renv: {}'.format(Renv))
    #------------------
    MStar = MStar * u.MSun
    LStar = u.LSun * ( MStar/u.MSun )**4  #L propto M**4?
    
    #-------------------------------
    #Parameters for the Pringle disc
    #-------------------------------
    MRate = MRate * u.MSun_yr
    RStar = 10*u.RSun * ( MStar/u.MSun )**0.8  #????
    
    #RStar = 26 * u.RSun * ( MStar/u.MSun )**0.27 * ( MRate / (1e-3*u.MSun_yr) )**0.41
    #from hosakawa 2009 relation for adiabatic accretion phase
    #LStar=  1e5*u.Lsun

    print('RStar:'.format(RStar))
    TStar = u.TSun * ( (LStar/u.LSun) / (RStar/u.RSun)**2 )**0.25
    Rd = Rdisc * u.au

    print ('RStar:', RStar/u.RSun,', LStar:', LStar/u.LSun, ', TStar:', TStar)
    
    #---------------
    #GRID Definition
    #---------------
    #Cubic grid, each edge ranges [-size, size] au.
    half_size_au = grid_config["half_size_au"]
    npoints = grid_config["npoints"]
    include_zero = grid_config.get("include_zero", True)
    grid_size = np.asarray(half_size_au) * u.au
    grid_npoints = np.asarray(npoints, dtype=int)

    GRID = Model.grid(grid_size, grid_npoints,rt_code="radmc3d", include_zero=include_zero)
    NPoints = GRID.NPoints #Final number of nodes in the grid
  
    #--------
    #DENSITY
    #--------
    Rho0 = Res.Rho0(MRate, Rd, MStar) #Density normalization value. 
    Arho = Arho0 #Disc-envelope density factor
    Renv = Renv * u.au #Envelope radius
    Cavity = cavity_ang * np.pi/180 #Cavity opening angle 
    density = Model.density_Env_Disc(RStar, Rd, Rho0, Arho, GRID, exp_disc=exp_disc, 
                                     discFlag = discFlag, envFlag = envFlag,
                                     renv_max = Renv, ang_cavity = Cavity, 
                                     average_around_Rd=np.median, rho_min_env=rho_min_env)

    #---------------------
    # MODEL TEMPERATURE
    #---------------------
    temperature = Model.temperature(TStar, Rd,T10Env, RStar, MStar, MRate, BT, density, GRID, p=p)
    #temperature=Model.temperature_Constant(density, GRID, discTemp = 2.5*const_T, envTemp = const_T, backTemp = 30.0)
    #Whitney et al. exponent is p=0.33 (in Keto & Zhang (2/(4+p)) where p<-1  )   
    #--------
    #VELOCITY
    #--------
    vel = Model.velocity(RStar, MStar, Rd, density, GRID)

#    #**********************
#    #WRITE RADMC-3D FILES
#    #**********************
    abundance = molec_abund+np.zeros(GRID.NPoints) #Optimize for the molecule
    gtdratio = Model.gastodust(gtd_ratio,GRID.NPoints)
    microturb = microturbulence +np.zeros(GRID.NPoints)
    prop = {'dens_H2': density.total,
        'dens_dust': 2*ct.mH * density.total * 1/gtdratio, #mass density # ct.mH -> u.amu
        'temp_dust': temperature.total, #30+np.zeros_like(density.total),
        #'temp_gas': temperature.total, 
        'velocity': [vel.x, vel.y, vel.z],
        'gtdratio': gtdratio,
        'microturbulence': microturb,
        'abundance': abundance}


    if diagnostic_plots:
        plot_ulrichdisk_diagnostics(GRID=GRID,prop=prop,density=density, MStar_msun=MStar/u.MSun, Rdisc_au=Rdisc,Renv_au=Renv / u.au,tag=diagnostic_tag,output_dir=diagnostic_output_dir,show=False)

    if prop_only: return GRID, prop, density
    
    radmc = rt.Radmc3d(GRID)
    
    wavelength_intervals = (radmc_config["wavelength_intervals_micron"]) 

    wavelength_divisions = (radmc_config["wavelength_divisions"])

    radmc.write_radmc3d_control(
        nphot=radmc_config["nphot"],
        incl_dust=radmc_config["incl_dust"],
        setthreads=radmc_config["threads"],
        incl_freefree=radmc_config["incl_freefree"],
        tgas_eq_tdust=radmc_config["tgas_eq_tdust"],
        modified_random_walk=radmc_config[
            "modified_random_walk"
        ]
    )
    
    
    #Threads=1 is the default for ray-traycing, >1 is useful for paralelized MCTherm
    radmc.write_amr_grid()
    radmc.write_dust_density(prop['dens_dust']) #Mass density
    radmc.write_dust_temperature(prop['temp_dust']) #Spherical radial plaw temperature produces artifacts near the disc outer radius in the line images
    radmc.write_gas_temperature(prop['temp_dust'])
    radmc.write_gas_velocity(prop['velocity'])
    radmc.write_microturbulence(prop['microturbulence'])
    radmc.write_stars(nstars=1, pos=[[0,0,0]], rstars = [u.RSun], mstars = [MStar], flux = [[-u.TSun]], #flux --> if negative, radmc assumes the input number as the blackbody temperature of the star 
                  lam = wavelength_intervals, nxx = wavelength_divisions) 
    radmc.write_wavelength_micron(lam = wavelength_intervals, nxx = wavelength_divisions) #lam --> wavelengths in microns, nxx --> number of divisions in between wavelengths
    nx,ny,nz = GRID.Nodes

    write_molecule_files(nx=nx,ny=ny,nz=nz, density=density, prop=prop, molec=molec)

    #plot_1d_props(GRID, prop, density, tag='')
    #-------
    #TIMING
    #-------
    print ('Ellapsed time for iterarion of create_model.py: %.3fs' % (time.time() - t0))
    print ('-------------------------------------------------\n-------------------------------------------------\n')

    

def Hamburguers(nmodel=0, MStar=20, MRate=5e-4, discFlag = True, Rdisc=300, Arho0=5, 
prop_only=False, molec='ch3oh', molec_abund=7.5e-6, p=0.5):

#-------
#DISC
#-------
    H0sf = 0.03 #Disc scale height factor (H0 = H0sf * RStar)
    Arho = Arho0 #Disc density factor
   
    t0 = time.time()

    print('\n')
    print('Passed paremeters are:')
    print('nmodel: {}'.format(nmodel))
    print('MStar: {}'.format(MStar))
    print('MRate: {}'.format(MRate))
    print('Rdisc: {}'.format(Rdisc))
    print('Arho0: {}'.format(Arho0))
    #print('Renv: {}'.format(Renv))
    #------------------
    MStar = MStar * u.MSun
    LStar = u.LSun * ( MStar/u.MSun )**4
    
    #-------------------------------
    #Parameters for the Pringle disc
    #-------------------------------
    MRate = MRate * u.MSun_yr
    RStar = u.RSun * ( MStar/u.MSun )**0.8
    print('RStar:'.format(RStar))
    TStar = u.TSun * ( (LStar/u.LSun) / (RStar/u.RSun)**2 )**0.25
    Rd = Rdisc * u.au

    print ('RStar:', RStar/u.RSun,', LStar:', LStar/u.LSun, ', TStar:', TStar)
    
    #---------------
    #GRID Definition
    #---------------
    #Cubic grid, each edge ranges [-size, size] au.

    sizex = sizey = sizez = 1648 * u.au
    Nx = Ny = Nz = 103 #Number of divisions for each axis
    GRID = Model.grid([sizex, sizey, sizez], [Nx, Ny, Nz], rt_code = 'radmc3d', include_zero = True)
    NPoints = GRID.NPoints #Final number of nodes in the grid
  

    #--------
    #DENSITY
    #--------
    Rho0 = Res.Rho0(MRate, Rd, MStar)
    print('#####%f######'%(Rho0))
    density = Model.density_Hamburgers(RStar, H0sf, Rd, Rho0, Arho, GRID,
                                    discFlag = True, rdisc_max = Rd*1.5, p=p)
    
    BT=5
    T10Env = 400
    temperature = Model.temperature_Hamburgers(TStar, RStar, MStar, MRate, Rd, T10Env, BT, density, GRID, 
                           p = 0.33, Tmin_disc = 30., Tmin_env = 30., inverted = False)
    

#TStar: stellar temperature
#T10Env: Envelope temperature at 10AU
#RStar: stellar radius
#MStar: stellar mass
#MRate: Mass accretion rate
#BT: Disc temperature factor
#p: Temperature power law exponent 
#GRID: [xList,yList,zList]
    
   
    #--------
    #VELOCITY
    #--------
    vel = Model.velocity(RStar, MStar, Rd, density, GRID)
#(IOnized gas)
    Model.PrintProperties(density, temperature, GRID, species='dens_ion')
    Model.PrintProperties(density, temperature, GRID, species='dens_e')
    
   #**********************
    #WRITE RADMC-3D FILES
    #**********************
    abundance = molec_abund+np.zeros(GRID.NPoints) #Optimize for the molecule
    gtdratio = Model.gastodust(100., GRID.NPoints)
    microturb = 100.+np.zeros(GRID.NPoints)
    prop = {'dens_H2': density.total,
        'dens_dust': 2*ct.mH * density.total * 1/gtdratio, #mass density # ct.mH -> u.amu
        'temp_dust': temperature.total, #30+np.zeros_like(density.total),
        #'temp_gas': temperature.total, 
        'velocity': [vel.x, vel.y, vel.z],
        'gtdratio': gtdratio,
        'microturbulence': microturb,
        'abundance': abundance}

    if prop_only: return GRID, prop, density
    
    radmc = rt.Radmc3d(GRID)
    wavelength_intervals = [1e-1,5e2,1e4] #[5e-3, 5e1, 1e4]
    wavelength_divisions = [20,20] 
    radmc.write_radmc3d_control(nphot=100000000, incl_dust=1, setthreads=8, incl_freefree=0, tgas_eq_tdust=1, modified_random_walk=1)
    radmc.write_amr_grid()
    radmc.write_dust_density(prop['dens_dust']) #Mass density
    radmc.write_dust_temperature(prop['temp_dust']) #Spherical radial plaw temperature produces artifacts near the disc outer radius in the line images
    radmc.write_gas_velocity(prop['velocity'])
    radmc.write_microturbulence(prop['microturbulence'])
    radmc.write_stars(nstars=1, pos=[[0,0,0]], rstars = [u.RSun], mstars = [MStar], flux = [[-u.TSun]], #flux --> if negative, radmc assumes the input number as the blackbody temperature of the star 
                  lam = wavelength_intervals, nxx = wavelength_divisions) 
    radmc.write_wavelength_micron(lam = wavelength_intervals, nxx = wavelength_divisions) #lam --> wavelengths in microns, nxx --> number of divisions in between wavelengths
    nx,ny,nz = GRID.Nodes
    
    write_molecule_files(nx=nx,ny=ny,nz=nz, density=density, prop=prop, molec=molec)

    
   
    #3D Points Distribution (weighting with density)
#-----------------------------------------------

    tag = 'Main'
    dens_plot = density.total / 1e6

    weight = 10*Rho0
    r = GRID.rRTP[0] / u.au #GRID.rRTP hosts [r, R, Theta, Phi] --> Polar GRID
    Plot_model.scatter3D(GRID, density.total, weight,
                         NRand = 4000, colordim = r, axisunit = u.au,
                         cmap = 'jet', colorscale = 'log',
                         colorlabel = r'${\rm log}_{10}(r [au])$',
                         output = '3Dpoints%s.png'%tag, show = False)

    #-------
    #TIMING
    #-------
    print ('Ellapsed time for iterarion of create_model.py: %.3fs' % (time.time() - t0))
    print ('-------------------------------------------------\n-------------------------------------------------\n')

    

def Hamburguers_piecewise(nmodel=0, MStar=20, MRate=1e-4, discFlag = True, Rdisc=150, Arho0=10, prop_only=False, molec='co' ,molec_abund=5e-7):


#def density_Hamburgers_piecewise(RStar, H0, R_list, p_list, rho0, GRID, RH_list = None,
#                               q_list = [0.5], rho_thres = 10.0, rho_min = 0.0, 
 #                                Rt = False):

#RStar: stellar radius
#H0: scaleheight normalization constant --> usually H0 = shFactor * R0
#R_list: List of polar limits, length (n,)
#p_list: List of powerlaws in R_list intervals, length (n-1,)
#rho0: density at R_list[0]
#Optionals:
#RH_list: List of limits for piecewise scaleheight, length (n,)
#q_list: List of powerlaws in RH_list intervals, length (n-1,)
#rho_thres: minimum reachable density by the model
#rho_min: background density
#Rt: radius where the disc tapering starts



#-------
#DISC
#-------
    H0sf = 0.03 #Disc scale height factor (H0 = H0sf * RStar)
    Arho = 5.25 #Disc density factor
   
    t0 = time.time()

    print('\n')
    print('Passed paremeters are:')
    print('nmodel: {}'.format(nmodel))
    print('MStar: {}'.format(MStar))
    print('MRate: {}'.format(MRate))
    print('Rdisc: {}'.format(Rdisc))
    print('Arho0: {}'.format(Arho0))
    #print('Renv: {}'.format(Renv))
    #------------------
    MStar = MStar * u.MSun
    LStar = u.LSun * ( MStar/u.MSun )**4
    
    #-------------------------------
    #Parameters for the Pringle disc
    #-------------------------------
    MRate = MRate * u.MSun_yr
    RStar = u.RSun * ( MStar/u.MSun )**0.8
    print('RStar:'.format(RStar))
    TStar = u.TSun * ( (LStar/u.LSun) / (RStar/u.RSun)**2 )**0.25
    Rd = Rdisc * u.au

    print ('RStar:', RStar/u.RSun,', LStar:', LStar/u.LSun, ', TStar:', TStar)
    
    #---------------
    #GRID Definition
    #---------------
    #Cubic grid, each edge ranges [-size, size] au.

    sizex = sizey = sizez = 4000 * u.au
    Nx = Ny = Nz = 80 #Number of divisions for each axis
    GRID = Model.grid([sizex, sizey, sizez], [Nx, Ny, Nz], rt_code = 'radmc3d', include_zero = True)
    NPoints = GRID.NPoints #Final number of nodes in the grid
  

    #--------
    #DENSITY
    #--------
    Rho0 = Res.Rho0(MRate, Rd, MStar)
    print('#####%f######'%(Rho0))
    density = Model.density_Hamburgers_piecewise(RStar, RStar*H0sf, [Rd/1000,Rd/3, 2*Rd/3,Rd],[1,0.5,0.2], Rho0, GRID)

    #---------------------
    # MODEL TEMPERATURE
    #---------------------
    # Ionized gas temperature

    t_e = 100 #K


    temperature = Model.temperature_Constant(density, GRID, discTemp=t_e, backTemp=2.725480)

    
   
    #--------
    #VELOCITY
    #--------
    vel = Model.velocity(RStar, MStar, Rd, density, GRID)
#(IOnized gas)
    Model.PrintProperties(density, temperature, GRID, species='dens_ion')
    Model.PrintProperties(density, temperature, GRID, species='dens_e')
    
    
    #-------------------------
    #ROTATION, VSYS, CENTERING
    #-------------------------

    #xc, yc, zc = [0.0,0.0,0.0]
    #CENTER = [xc, yc, zc] #New center of the region in the global grid
    #newProperties = Model.ChangeGeometry(GRID, center = CENTER,  vel = vel,
    	      	 	             #rot_dict = {'angles': [0*(np.pi/2)*(60./90)], 'axis': ['x'] })
    #GRID.XYZ = newProperties.newXYZ #XYZ redefinition
    #vel.x, vel.y, vel.z = newProperties.newVEL #vels redefinition

    #rot_dict = {'angles': [(np.pi/2)*(135./90),(np.pi/2)*(60./90)], 'axis': ['y','x'] })


    #**********************
    #WRITE RADMC-3D FILES
    #**********************
    abundance = molec_abund+np.zeros(GRID.NPoints) #Optimize for the molecule
    gtdratio = Model.gastodust(100., GRID.NPoints)
    microturb = 100.+np.zeros(GRID.NPoints)
    prop = {'dens_H2': density.total,
        'dens_dust': 2*ct.mH * density.total * 1/gtdratio, #mass density # ct.mH -> u.amu
        'temp_dust': temperature.total, #30+np.zeros_like(density.total),
        #'temp_gas': temperature.total, 
        'velocity': [vel.x, vel.y, vel.z],
        'gtdratio': gtdratio,
        'microturbulence': microturb,
        'abundance': abundance}

    if prop_only: return GRID, prop, density
    
    radmc = rt.Radmc3d(GRID)
    wavelength_intervals = [1e-1,5e2,1e4] #[5e-3, 5e1, 1e4]
    wavelength_divisions = [20,20] 
    radmc.write_radmc3d_control(nphot=100000000, incl_dust=1, setthreads=8, incl_freefree=0, tgas_eq_tdust=1, modified_random_walk=1)
    radmc.write_amr_grid()
    radmc.write_dust_density(prop['dens_dust']) #Mass density
    radmc.write_dust_temperature(prop['temp_dust']) #Spherical radial plaw temperature produces artifacts near the disc outer radius in the line images
    radmc.write_gas_velocity(prop['velocity'])
    radmc.write_microturbulence(prop['microturbulence'])
    radmc.write_stars(nstars=1, pos=[[0,0,0]], rstars = [u.RSun], mstars = [MStar], flux = [[-u.TSun]], #flux --> if negative, radmc assumes the input number as the blackbody temperature of the star 
                  lam = wavelength_intervals, nxx = wavelength_divisions) 
    radmc.write_wavelength_micron(lam = wavelength_intervals, nxx = wavelength_divisions) #lam --> wavelengths in microns, nxx --> number of divisions in between wavelengths
    nx,ny,nz = GRID.Nodes
    
    write_molecule_files(nx=nx,ny=ny,nz=nz, density=density, prop=prop, molec=molec)

    
   
    #3D Points Distribution (weighting with density)
#-----------------------------------------------

    tag = 'Main'
    dens_plot = density.total / 1e6

    weight = 10*Rho0
    r = GRID.rRTP[0] / u.au #GRID.rRTP hosts [r, R, Theta, Phi] --> Polar GRID
    Plot_model.scatter3D(GRID, density.total, weight,
                         NRand = 4000, colordim = r, axisunit = u.au,
                         cmap = 'jet', colorscale = 'log',
                         colorlabel = r'${\rm log}_{10}(r [au])$',
                         output = '3Dpoints%s.png'%tag, show = False)

    #-------
    #TIMING
    #-------
    print ('Ellapsed time for iterarion of create_model.py: %.3fs' % (time.time() - t0))
    print ('-------------------------------------------------\n-------------------------------------------------\n')