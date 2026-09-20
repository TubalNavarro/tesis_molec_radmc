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
        
def UlrichDisk(nmodel, *, MStar, MRate, Rdisc, Arho0, Renv, cavity_ang, exp_disc, molec_abund, BT, T10Env, p, lum_bol, molec, grid_config, model_config, radmc_config, prop_only=False, diagnostic_plots=False, diagnostic_tag="Main", diagnostic_output_dir="."):
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

    #LStar = u.LSun * ( MStar/u.MSun )**4  #L propto M**4?
    LStar = u.LSun *lum_bol

    #-------------------------------
    #Parameters for the Pringle disc
    #-------------------------------
    MRate = MRate * u.MSun_yr
    #RStar = u.RSun * ( MStar/u.MSun )**0.8  
    
    RStar = 26 * u.RSun * ( MStar/u.MSun )**0.27 * ( MRate / (1e-3*u.MSun_yr) )**0.41
    #from hosakawa 2009 relation for adiabatic accretion phase
   

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
    temperature = Model.temperature(TStar, Rd,T10Env, RStar, MStar, MRate, BT, density, GRID, p=p, Tmin_env=22)
    #temperature=Model.temperature_Constant(density, GRID, discTemp = 2.5*const_T, envTemp = const_T, backTemp = 30.0)
    #Whitney et al. exponent is p=0.33 (optically thin) (in Keto & Zhang (2/(4+p)) where p<-1 is for the model disc+env  )   
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

    


def _require_model_configs(grid_config, model_config, radmc_config):
    """Validate the configuration blocks shared by all model builders."""
    if grid_config is None:
        raise ValueError("grid_config must be provided")
    if model_config is None:
        raise ValueError("model_config must be provided")
    if radmc_config is None:
        raise ValueError("radmc_config must be provided")


def _make_configured_grid(grid_config):
    """Create the sf3dmodels grid from the project JSON."""
    half_size_au = grid_config["half_size_au"]
    npoints = grid_config["npoints"]
    include_zero = grid_config.get("include_zero", True)

    grid_size = np.asarray(half_size_au, dtype=float) * u.au
    grid_npoints = np.asarray(npoints, dtype=int)

    return Model.grid(
        grid_size,
        grid_npoints,
        rt_code="radmc3d",
        include_zero=include_zero,
    )


def _hamburger_stellar_properties(MStar_msun, MRate_msunyr, lum_bol, model_config):
    """
    Return MStar, MRate, RStar and TStar in the native sf3dmodels units.

    Two RStar prescriptions are supported so old Hamburger runs can be
    reproduced while allowing the same Hosokawa-like prescription used by
    UlrichDisk:
      - legacy_mass_powerlaw: RStar/Rsun = (MStar/Msun)^0.8
      - hosokawa_accretion:   current UlrichDisk prescription
    """
    MStar = float(MStar_msun) * u.MSun
    MRate = float(MRate_msunyr) * u.MSun_yr

    radius_mode = model_config.get(
        "stellar_radius_mode",
        "legacy_mass_powerlaw",
    ).lower()

    if radius_mode == "legacy_mass_powerlaw":
        RStar = u.RSun * (MStar / u.MSun) ** 0.8
    elif radius_mode == "hosokawa_accretion":
        RStar = (
            26
            * u.RSun
            * (MStar / u.MSun) ** 0.27
            * (MRate / (1e-3 * u.MSun_yr)) ** 0.41
        )
    else:
        raise ValueError(
            "Unknown stellar_radius_mode "
            f"'{radius_mode}'. Use 'legacy_mass_powerlaw' "
            "or 'hosokawa_accretion'."
        )

    LStar = float(lum_bol) * u.LSun
    TStar = u.TSun * (
        (LStar / u.LSun) / (RStar / u.RSun) ** 2
    ) ** 0.25

    return MStar, MRate, RStar, TStar


def _write_radmc_model(
    GRID,
    prop,
    density,
    MStar,
    molec,
    radmc_config,
):
    """Write the RADMC-3D inputs shared by the three model families."""
    radmc = rt.Radmc3d(GRID)

    wavelength_intervals = radmc_config[
        "wavelength_intervals_micron"
    ]
    wavelength_divisions = radmc_config[
        "wavelength_divisions"
    ]

    radmc.write_radmc3d_control(
        nphot=radmc_config["nphot"],
        incl_dust=radmc_config["incl_dust"],
        setthreads=radmc_config["threads"],
        incl_freefree=radmc_config["incl_freefree"],
        tgas_eq_tdust=radmc_config["tgas_eq_tdust"],
        modified_random_walk=radmc_config[
            "modified_random_walk"
        ],
    )

    radmc.write_amr_grid()
    radmc.write_dust_density(prop["dens_dust"])
    radmc.write_dust_temperature(prop["temp_dust"])
    radmc.write_gas_temperature(prop["temp_dust"])
    radmc.write_gas_velocity(prop["velocity"])
    radmc.write_microturbulence(prop["microturbulence"])

    # Preserve the stellar write-out used by the current UlrichDisk routine.
    radmc.write_stars(
        nstars=1,
        pos=[[0, 0, 0]],
        rstars=[u.RSun],
        mstars=[MStar],
        flux=[[-u.TSun]],
        lam=wavelength_intervals,
        nxx=wavelength_divisions,
    )
    radmc.write_wavelength_micron(
        lam=wavelength_intervals,
        nxx=wavelength_divisions,
    )

    nx, ny, nz = GRID.Nodes
    write_molecule_files(
        nx=nx,
        ny=ny,
        nz=nz,
        density=density,
        prop=prop,
        molec=molec,
    )


def _make_prop(
    GRID,
    density,
    temperature,
    vel,
    molec_abund,
    model_config,
):
    """Assemble the property dictionary expected by RADMC-3D."""
    gtd_ratio = model_config["gas_to_dust_ratio"]
    microturbulence = model_config["microturbulence_ms"]

    abundance = float(molec_abund) + np.zeros(GRID.NPoints)
    gtdratio = Model.gastodust(gtd_ratio, GRID.NPoints)
    microturb = microturbulence + np.zeros(GRID.NPoints)

    return {
        "dens_H2": density.total,
        "dens_dust": 2 * ct.mH * density.total * 1 / gtdratio,
        "temp_dust": temperature.total,
        "velocity": [vel.x, vel.y, vel.z],
        "gtdratio": gtdratio,
        "microturbulence": microturb,
        "abundance": abundance,
    }


def _resolve_piecewise_radii(section, Rdisc_au, prefix="R"):
    """
    Resolve a piecewise-radius list from either absolute AU values or
    fractions of Rdisc. Returns native sf3dmodels length units.
    """
    abs_key = f"{prefix}_breaks_au"
    frac_key = f"{prefix}_breaks_fraction"

    if abs_key in section:
        radii_au = np.asarray(section[abs_key], dtype=float)
    elif frac_key in section:
        radii_au = (
            np.asarray(section[frac_key], dtype=float)
            * float(Rdisc_au)
        )
    else:
        raise ValueError(
            f"Piecewise section needs '{abs_key}' "
            f"or '{frac_key}'."
        )

    if radii_au.ndim != 1 or radii_au.size < 2:
        raise ValueError("Piecewise radii must contain at least 2 limits.")
    if np.any(np.diff(radii_au) <= 0):
        raise ValueError("Piecewise radii must be strictly increasing.")
    if radii_au[0] <= 0:
        raise ValueError("The first piecewise radius must be > 0.")

    return radii_au * u.au


def _validate_piecewise_exponents(radii, exponents, label):
    exponents = list(exponents)
    if len(exponents) != len(radii) - 1:
        raise ValueError(
            f"{label}: expected {len(radii)-1} exponents "
            f"for {len(radii)} radius limits, got {len(exponents)}."
        )
    return exponents


def _piecewise_velocity(
    density,
    GRID,
    MStar,
    Rdisc_au,
    model_config,
    vphi_factor=1.0,
    vr_factor=0.0,
    vr0_kms=None,
):
    """
    Build the velocity field with sf3dmodels.velocity_piecewise.

    vphi_factor scales the Keplerian speed at the first rotational break.
    vr_factor scales the free-fall speed at the first radial break;
    positive vr_factor means inward motion.
    """
    cfg = model_config.get("velocity_piecewise", {})
    rotation = cfg.get("rotation", {})
    infall = cfg.get("infall", {})

    R_list = pR_list = v0R = None
    r_list = pr_list = v0r = None

    if rotation.get("enabled", True):
        R_list = _resolve_piecewise_radii(
            rotation,
            Rdisc_au,
            prefix="R",
        )
        pR_list = _validate_piecewise_exponents(
            R_list,
            rotation.get("p_list", []),
            "velocity_piecewise.rotation",
        )

        if "v0_kms" in rotation:
            vphi0 = float(rotation["v0_kms"]) * 1e3
        else:
            vphi0 = (
                float(vphi_factor)
                * np.sqrt(G.value * MStar / R_list[0])
            )

        v0R = [0.0, 0.0, vphi0]

    if infall.get("enabled", False):
        r_list = _resolve_piecewise_radii(
            infall,
            Rdisc_au,
            prefix="r",
        )
        pr_list = _validate_piecewise_exponents(
            r_list,
            infall.get("p_list", []),
            "velocity_piecewise.infall",
        )

        if "v0_kms" in infall:
            # Fixed value specified directly in the project JSON.
            vr0 = -abs(float(infall["v0_kms"])) * 1e3

        elif vr0_kms is not None:
            # Free scalar parameter supplied by test_pars / MCMC CSV.
            # Positive vr0_kms means inward motion.
            vr0 = -abs(float(vr0_kms)) * 1e3

        else:
            # Backward-compatible mode:
            # vr_factor is a fraction of the free-fall velocity.
            vr0 = (
                -abs(float(vr_factor))
                * np.sqrt(2.0 * G.value * MStar / r_list[0])
            )

        v0r = [vr0, 0.0, 0.0]

    if R_list is None and r_list is None:
        raise ValueError(
            "velocity_mode='piecewise' but both rotation and infall "
            "are disabled."
        )

    return Model.velocity_piecewise(
        density,
        GRID,
        R_list=R_list,
        pR_list=pR_list,
        v0R=v0R,
        r_list=r_list,
        pr_list=pr_list,
        v0r=v0r,
    )


def _hamburger_velocity(
    RStar,
    MStar,
    Rd,
    density,
    GRID,
    Rdisc_au,
    model_config,
    vphi_factor=1.0,
    vr_factor=0.0,
    vr0_kms=None,
    default_mode="standard",
):
    velocity_mode = model_config.get(
        "velocity_mode",
        default_mode,
    ).lower()

    if velocity_mode == "standard":
        return Model.velocity(
            RStar,
            MStar,
            Rd,
            density,
            GRID,
        )

    if velocity_mode == "piecewise":
        return _piecewise_velocity(
            density=density,
            GRID=GRID,
            MStar=MStar,
            Rdisc_au=Rdisc_au,
            model_config=model_config,
            vphi_factor=vphi_factor,
            vr_factor=vr_factor,
            vr0_kms=vr0_kms,
        )

    raise ValueError(
        f"Unknown velocity_mode '{velocity_mode}'. "
        "Use 'standard' or 'piecewise'."
    )


def Hamburguers(
    nmodel=0,
    *,
    MStar=20,
    MRate=5e-4,
    Rdisc=300,
    Arho0=5,
    H0_factor=0.03,
    p_density=2.25,
    q_density=0.5,
    BT=5.0,
    T10Env=400.0,
    p_temp=0.33,
    Tmin_disc=30.0,
    Tmin_env=30.0,
    molec_abund=7.5e-6,
    vphi_factor=1.0,
    vr_factor=0.0,
    vr0_kms=None,
    lum_bol,
    molec,
    grid_config,
    model_config,
    radmc_config,
    prop_only=False,
    diagnostic_plots=True,
    diagnostic_tag="Main",
    diagnostic_output_dir=".",
):
    """
    Config-driven Hamburger disk model.

    All run/environment settings come from the same JSON structure used by
    UlrichDisk. Physical scalar parameters can therefore live in a model-
    specific CSV and be fitted by the same MCMC driver.
    """
    t0 = time.time()
    _require_model_configs(
        grid_config,
        model_config,
        radmc_config,
    )

    discFlag = model_config.get("disc", True)
    envFlag = model_config.get("envelope", False)

    if not discFlag:
        raise ValueError("Hamburguers requires model_options.disc=true.")
    if envFlag:
        raise ValueError(
            "density_Hamburgers is a disk-only density law; "
            "set model_options.envelope=false."
        )

    print("\n")
    print("Passed parameters are:")
    print(f"nmodel: {nmodel}")
    print(f"MStar: {MStar}")
    print(f"MRate: {MRate}")
    print(f"Rdisc: {Rdisc}")
    print(f"Arho0: {Arho0}")
    print(f"H0_factor: {H0_factor}")
    print(f"p_density: {p_density}")
    print(f"q_density: {q_density}")

    MStar_si, MRate_si, RStar, TStar = (
        _hamburger_stellar_properties(
            MStar,
            MRate,
            lum_bol,
            model_config,
        )
    )
    Rd = float(Rdisc) * u.au

    print(
        "RStar:",
        RStar / u.RSun,
        ", TStar:",
        TStar,
    )

    GRID = _make_configured_grid(grid_config)

    Rho0 = Res.Rho0(
        MRate_si,
        Rd,
        MStar_si,
    )

    rho_thres = model_config.get("rho_thres", 10.0)
    rho_min = model_config.get("rho_min", 1.0)

    rt_au = model_config.get("Rt_au")
    Rt = False if rt_au is None else float(rt_au) * u.au

    if "rdisc_max_au" in model_config:
        rdisc_max = float(model_config["rdisc_max_au"]) * u.au
    else:
        rdisc_max = (
            float(model_config.get("rdisc_max_factor", 1.0))
            * Rd
        )

    density = Model.density_Hamburgers(
        RStar,
        float(H0_factor),
        Rd,
        Rho0,
        float(Arho0),
        GRID,
        p=float(p_density),
        q=float(q_density),
        rho_thres=float(rho_thres),
        rho_min=float(rho_min),
        Rt=Rt,
        discFlag=True,
        rdisc_max=rdisc_max,
    )

    temperature = Model.temperature_Hamburgers(
        TStar,
        RStar,
        MStar_si,
        MRate_si,
        Rd,
        float(T10Env),
        float(BT),
        density,
        GRID,
        p=float(p_temp),
        Tmin_disc=float(Tmin_disc),
        Tmin_env=float(Tmin_env),
        inverted=bool(model_config.get("inverted_temperature", False)),
    )

    vel = _hamburger_velocity(
        RStar,
        MStar_si,
        Rd,
        density,
        GRID,
        Rdisc,
        model_config,
        vphi_factor=vphi_factor,
        vr_factor=vr_factor,
        vr0_kms=vr0_kms,
        default_mode="standard",
    )

    prop = _make_prop(
        GRID,
        density,
        temperature,
        vel,
        molec_abund,
        model_config,
    )

    if diagnostic_plots:
        plot_ulrichdisk_diagnostics(
            GRID=GRID,
            prop=prop,
            density=density,
            MStar_msun=MStar_si / u.MSun,
            Rdisc_au=float(Rdisc),
            Renv_au=None,
            tag=diagnostic_tag,
            output_dir=diagnostic_output_dir,
            show=False,
        )

    if prop_only:
        return GRID, prop, density

    _write_radmc_model(
        GRID,
        prop,
        density,
        MStar_si,
        molec,
        radmc_config,
    )

    print(
        "Ellapsed time for iteration of create_model.py: "
        f"{time.time() - t0:.3f}s"
    )
    print(
        "-------------------------------------------------\n"
        "-------------------------------------------------\n"
    )


def Hamburguers_piecewise(
    nmodel=0,
    *,
    MStar=20,
    MRate=1e-4,
    Rdisc=150,
    Arho0=10,
    H0_factor=0.03,
    BT=5.0,
    T10Env=400.0,
    p_temp=0.33,
    Tmin_disc=30.0,
    Tmin_env=30.0,
    molec_abund=5e-7,
    vphi_factor=1.0,
    vr_factor=0.0,
    lum_bol,
    molec,
    grid_config,
    model_config,
    radmc_config,
    prop_only=False,
    diagnostic_plots=True,
    diagnostic_tag="Main",
    diagnostic_output_dir=".",
):
    """
    Config-driven piecewise Hamburger disk model.

    The piecewise radius limits and exponents are stored in model_options
    because they are arrays. Scalar normalizations/factors remain available
    in the CSV and can be fitted by MCMC.
    """
    t0 = time.time()
    _require_model_configs(
        grid_config,
        model_config,
        radmc_config,
    )

    discFlag = model_config.get("disc", True)
    envFlag = model_config.get("envelope", False)

    if not discFlag:
        raise ValueError(
            "Hamburguers_piecewise requires model_options.disc=true."
        )
    if envFlag:
        raise ValueError(
            "density_Hamburgers_piecewise is disk-only; "
            "set model_options.envelope=false."
        )

    print("\n")
    print("Passed parameters are:")
    print(f"nmodel: {nmodel}")
    print(f"MStar: {MStar}")
    print(f"MRate: {MRate}")
    print(f"Rdisc: {Rdisc}")
    print(f"Arho0: {Arho0}")
    print(f"H0_factor: {H0_factor}")
    print(f"vphi_factor: {vphi_factor}")
    print(f"vr_factor: {vr_factor}")

    MStar_si, MRate_si, RStar, TStar = (
        _hamburger_stellar_properties(
            MStar,
            MRate,
            lum_bol,
            model_config,
        )
    )
    Rd = float(Rdisc) * u.au

    GRID = _make_configured_grid(grid_config)

    Rho0 = Res.Rho0(
        MRate_si,
        Rd,
        MStar_si,
    )

    density_cfg = model_config.get("density_piecewise")
    if density_cfg is None:
        raise ValueError(
            "Hamburguers_piecewise needs "
            "model_options.density_piecewise."
        )

    R_list = _resolve_piecewise_radii(
        density_cfg,
        Rdisc,
        prefix="R",
    )
    p_list = _validate_piecewise_exponents(
        R_list,
        density_cfg.get("p_list", []),
        "density_piecewise",
    )

    RH_list = None
    if (
        "RH_breaks_au" in density_cfg
        or "RH_breaks_fraction" in density_cfg
    ):
        # _resolve_piecewise_radii expects a one-letter prefix, so handle
        # the scale-height limits explicitly here.
        if "RH_breaks_au" in density_cfg:
            RH_list = (
                np.asarray(
                    density_cfg["RH_breaks_au"],
                    dtype=float,
                )
                * u.au
            )
        else:
            RH_list = (
                np.asarray(
                    density_cfg["RH_breaks_fraction"],
                    dtype=float,
                )
                * float(Rdisc)
                * u.au
            )
        if np.any(np.diff(RH_list) <= 0):
            raise ValueError(
                "density_piecewise RH limits must be strictly increasing."
            )

    q_list = density_cfg.get("q_list", [0.5])
    if RH_list is not None:
        q_list = _validate_piecewise_exponents(
            RH_list,
            q_list,
            "density_piecewise scale height",
        )

    rt_au = model_config.get("Rt_au")
    Rt = False if rt_au is None else float(rt_au) * u.au

    density = Model.density_Hamburgers_piecewise(
        RStar,
        float(H0_factor) * RStar,
        R_list,
        p_list,
        float(Arho0) * Rho0,
        GRID,
        RH_list=RH_list,
        q_list=q_list,
        rho_thres=float(model_config.get("rho_thres", 10.0)),
        rho_min=float(model_config.get("rho_min", 1.0)),
        Rt=Rt,
    )

    temperature = Model.temperature_Hamburgers(
        TStar,
        RStar,
        MStar_si,
        MRate_si,
        Rd,
        float(T10Env),
        float(BT),
        density,
        GRID,
        p=float(p_temp),
        Tmin_disc=float(Tmin_disc),
        Tmin_env=float(Tmin_env),
        inverted=bool(model_config.get("inverted_temperature", False)),
    )

    vel = _hamburger_velocity(
        RStar,
        MStar_si,
        Rd,
        density,
        GRID,
        Rdisc,
        model_config,
        vphi_factor=vphi_factor,
        vr_factor=vr_factor,
        default_mode="piecewise",
    )

    prop = _make_prop(
        GRID,
        density,
        temperature,
        vel,
        molec_abund,
        model_config,
    )

    if diagnostic_plots:
        plot_ulrichdisk_diagnostics(
            GRID=GRID,
            prop=prop,
            density=density,
            MStar_msun=MStar_si / u.MSun,
            Rdisc_au=float(Rdisc),
            Renv_au=None,
            tag=diagnostic_tag,
            output_dir=diagnostic_output_dir,
            show=False,
        )

    if prop_only:
        return GRID, prop, density

    _write_radmc_model(
        GRID,
        prop,
        density,
        MStar_si,
        molec,
        radmc_config,
    )

    print(
        "Ellapsed time for iteration of create_model.py: "
        f"{time.time() - t0:.3f}s"
    )
    print(
        "-------------------------------------------------\n"
        "-------------------------------------------------\n"
    )


_MODEL_ALIASES = {
    "ulrich": "ulrich",
    "ulrichdisk": "ulrich",
    "ulrich_disk": "ulrich",
    "hamburgers": "hamburgers",
    "hamburger": "hamburgers",
    "hamburguers": "hamburgers",
    "hamburguers_disk": "hamburgers",
    "hamburgers_piecewise": "hamburgers_piecewise",
    "hamburger_piecewise": "hamburgers_piecewise",
    "hamburguers_piecewise": "hamburgers_piecewise",
}


def get_model_function(model_type):
    """Return the configured yso_models builder for a model.type string."""
    key = str(model_type).strip().lower()
    canonical = _MODEL_ALIASES.get(key)

    if canonical is None:
        allowed = sorted(
            {
                "ulrich",
                "hamburgers",
                "hamburgers_piecewise",
            }
        )
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available models: {allowed}"
        )

    registry = {
        "ulrich": UlrichDisk,
        "hamburgers": Hamburguers,
        "hamburgers_piecewise": Hamburguers_piecewise,
    }
    return registry[canonical]
