import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as ct

from scipy.special import gamma, gammaincc


# ============================================================
# CONFIGURACIÓN
# ============================================================

ENERGY_FILE = "c032504.egy"

# Temperaturas donde probaremos explícitamente la convergencia
T_CONVERGENCE = np.logspace(
    np.log10(10.0),
    np.log10(1.0e5),
    20
) * u.K

# Mallado de temperatura para Q(T)
T_Q = np.logspace(
    np.log10(2.725),
    np.log10(1.0e5),
    500
) * u.K

# Tolerancia para considerar convergencia de Q_N
CONVERGENCE_TOL = 1.0e-3  # 0.1 %

# Última fracción de niveles usada como diagnóstico
TAIL_FRACTION_LEVELS = 0.10

# Región de energía utilizada para ajustar
#
# G(E) = A E^beta
#
# Evitamos deliberadamente la parte final del archivo,
# donde puede existir truncamiento.
FIT_MIN_FRACTION = 0.10
FIT_MAX_FRACTION = 0.60


# ============================================================
# VALORES TABULADOS DE Q(T)
# ============================================================

temps_table = np.array([
    2.725,
    5.0,
    9.375,
    18.75,
    37.5,
    75.0,
    150.0,
    225.0,
    300.0,
    500.0,
    1000.0
])

Qvals_table = 10**np.array([
    1.0752,
    1.4268,
    1.8931,
    2.4393,
    2.9642,
    3.4660,
    3.9890,
    4.3220,
    4.5685,
    5.0092,
    5.5413
])


# ============================================================
# LEER ARCHIVO DE ENERGÍAS
# ============================================================

def leer_tabla(archivo):
    """
    Lee el archivo .egy.

    Se asume una estructura del tipo:

        energia ... : numeros_cuanticos

    La primera columna antes de ':' se interpreta como energía
    en cm^-1.
    """

    filas = []

    with open(archivo, "r") as f:

        for linea in f:

            linea = linea.strip()

            if not linea:
                continue

            if ":" not in linea:
                continue

            izquierda, derecha = linea.split(":", 1)

            # Energía en cm^-1
            energia = float(
                izquierda.split()[0]
            )

            # Números cuánticos
            valores = [
                int(v)
                for v in derecha.split()
            ]

            filas.append(
                [energia] + valores
            )

    return np.array(
        filas,
        dtype=float
    )


# ============================================================
# LEER DATOS
# ============================================================

data = leer_tabla(
    ENERGY_FILE
)

print()
print("=" * 80)
print("ARCHIVO DE NIVELES")
print("=" * 80)

print(
    f"Archivo: {ENERGY_FILE}"
)

print(
    f"Número de niveles leídos: {len(data)}"
)


# ============================================================
# CONVERTIR ENERGÍAS A E/k_B EN K
# ============================================================

E = (
    (data[:, 0] / u.cm).to(1 / u.m)
    * ct.h
    * ct.c
    / ct.k_B
).to(u.K)


# ============================================================
# DEGENERACIÓN
#
# Mantengo exactamente tu expresión:
#
# g = (2J + 1) * 4
# ============================================================

J = data[:, 1]

g = (
    2.0 * J + 1.0
) * 4.0


# ============================================================
# ORDENAR LOS NIVELES POR ENERGÍA
# ============================================================

orden = np.argsort(
    E.value
)

E = E[orden]
g = g[orden]
data = data[orden]

Evals = E.to_value(u.K)

Nlevels = len(Evals)

print(
    f"E_min/k = {Evals[0]:.6f} K"
)

print(
    f"E_max/k = {Evals[-1]:.6f} K"
)

print()


# ============================================================
# FUNCIÓN DE PARTICIÓN
# ============================================================

def Q_acumulada(T):
    """
    Calcula:

        Q_N(T) =
        sum_{i=1}^{N} g_i exp(-E_i/T)

    para todos los valores de N.
    """

    Tval = u.Quantity(
        T
    ).to_value(u.K)

    contribuciones = (
        g
        * np.exp(
            -Evals / Tval
        )
    )

    return np.cumsum(
        contribuciones
    )


def Q_known(T):
    """
    Función de partición usando todos los niveles explícitos
    presentes en el archivo.
    """

    return Q_acumulada(
        T
    )[-1]


# ============================================================
# AJUSTAR NÚMERO ACUMULADO DE ESTADOS
# ============================================================

def estimate_density_of_states_cumulative(
    Evals,
    g,
    fit_min_fraction=0.10,
    fit_max_fraction=0.60
):
    """
    Ajusta el número acumulado de estados ponderado:

        G(E) = sum_{Ei <= E} g_i

    mediante:

        G(E) = A E^beta

    Entonces:

        rho(E) = dG/dE
               = A beta E^(beta - 1)

    De modo que:

        C = A beta
        alpha = beta - 1

    y:

        rho(E) = C E^alpha.
    """

    idx = np.argsort(
        Evals
    )

    E_sorted = np.asarray(
        Evals
    )[idx]

    g_sorted = np.asarray(
        g
    )[idx]

    # Suma acumulada ponderada por degeneración
    Gcum = np.cumsum(
        g_sorted
    )

    Emin = np.min(
        E_sorted
    )

    Emax = np.max(
        E_sorted
    )

    # Región de ajuste
    Efit_min = (
        Emin
        + fit_min_fraction
        * (Emax - Emin)
    )

    Efit_max = (
        Emin
        + fit_max_fraction
        * (Emax - Emin)
    )

    mask_fit = (
        (E_sorted >= Efit_min)
        &
        (E_sorted <= Efit_max)
        &
        (E_sorted > 0)
        &
        (Gcum > 0)
    )

    E_fit = E_sorted[
        mask_fit
    ]

    G_fit = Gcum[
        mask_fit
    ]

    if len(E_fit) < 10:

        raise RuntimeError(
            "No hay suficientes niveles "
            "en la región seleccionada "
            "para ajustar G(E)."
        )

    # --------------------------------------------------------
    # Ajuste log-log:
    #
    # log G =
    # log A + beta log E
    # --------------------------------------------------------

    coeff = np.polyfit(
        np.log(E_fit),
        np.log(G_fit),
        1
    )

    beta = coeff[0]

    logA = coeff[1]

    A = np.exp(
        logA
    )

    # --------------------------------------------------------
    # Derivada:
    #
    # rho(E) =
    # A beta E^(beta-1)
    # --------------------------------------------------------

    alpha = (
        beta - 1.0
    )

    C = (
        A * beta
    )


    # --------------------------------------------------------
    # Calidad del ajuste en log-log
    # --------------------------------------------------------

    logG_model = (
        np.log(A)
        + beta * np.log(E_fit)
    )

    residuals = (
        np.log(G_fit)
        - logG_model
    )

    ss_res = np.sum(
        residuals**2
    )

    ss_tot = np.sum(
        (
            np.log(G_fit)
            - np.mean(np.log(G_fit))
        )**2
    )

    if ss_tot > 0:

        R2 = (
            1.0
            - ss_res / ss_tot
        )

    else:

        R2 = np.nan


    print()
    print("=" * 80)
    print("AJUSTE ACUMULADO DE ESTADOS")
    print("=" * 80)

    print(
        f"G(E) = {A:.6e} E^{beta:.6f}"
    )

    print(
        f"rho(E) = {C:.6e} E^{alpha:.6f}"
    )

    print(
        f"beta = {beta:.6f}"
    )

    print(
        f"alpha = {alpha:.6f}"
    )

    print(
        f"Comportamiento asintótico: "
        f"Q(T) ~ T^{beta:.6f}"
    )

    print(
        f"Región de ajuste: "
        f"{Efit_min:.2f} -- "
        f"{Efit_max:.2f} K"
    )

    print(
        f"R^2 en log-log = {R2:.6f}"
    )

    print()


    return (
        E_sorted,
        Gcum,
        A,
        beta,
        C,
        alpha,
        mask_fit,
        R2
    )


# ============================================================
# REALIZAR AJUSTE
# ============================================================

(
    E_sorted,
    Gcum,
    A_states,
    beta_states,
    C_rho,
    alpha_rho,
    mask_fit_states,
    R2_states
) = estimate_density_of_states_cumulative(
    Evals,
    g,
    fit_min_fraction=FIT_MIN_FRACTION,
    fit_max_fraction=FIT_MAX_FRACTION
)


# ============================================================
# ESTIMAR PARTE FALTANTE DE Q
# ============================================================

def estimate_Q_complete(
    T,
    Evals,
    g,
    C,
    alpha
):
    """
    Calcula:

        Q_known
        Q_missing
        Q_estimated

    La densidad de estados extrapolada es:

        rho(E) = C E^alpha

    Para E > Emax:

        Q_missing =
        integral_Emax^infinity
        C E^alpha exp(-E/T) dE

    cuya solución es:

        Q_missing =
        C T^(alpha+1)
        Gamma(alpha+1, Emax/T).
    """

    Tval = u.Quantity(
        T
    ).to_value(u.K)

    Emax = np.max(
        Evals
    )

    # --------------------------------------------------------
    # Función de partición conocida
    # --------------------------------------------------------

    Qk = np.sum(
        g
        * np.exp(
            -Evals / Tval
        )
    )

    # --------------------------------------------------------
    # Exponente
    # --------------------------------------------------------

    a = (
        alpha + 1.0
    )

    if a <= 0:

        print(
            "ADVERTENCIA:"
        )

        print(
            f"alpha = {alpha:.6f}"
        )

        print(
            f"alpha + 1 = {a:.6f}"
        )

        print(
            "No se puede realizar "
            "una extrapolación física."
        )

        return (
            Qk,
            np.nan,
            np.nan
        )


    # --------------------------------------------------------
    # Gamma incompleta superior
    #
    # scipy:
    #
    # gammaincc(a,x) =
    # Gamma(a,x) / Gamma(a)
    # --------------------------------------------------------

    Gamma_upper = (
        gamma(a)
        * gammaincc(
            a,
            Emax / Tval
        )
    )

    # --------------------------------------------------------
    # Parte faltante
    # --------------------------------------------------------

    Qmiss = (
        C
        * Tval**a
        * Gamma_upper
    )

    Qest = (
        Qk + Qmiss
    )

    return (
        Qk,
        Qmiss,
        Qest
    )


# ============================================================
# CONVERGENCIA Q_N CON NÚMERO DE NIVELES
# ============================================================

print()
print("=" * 115)
print("CONVERGENCIA CON NÚMERO DE NIVELES")
print("=" * 115)

print(
    f"{'T [K]':>12}"
    f"{'Q known':>18}"
    f"{'Nconv':>12}"
    f"{'Nconv/N':>12}"
    f"{'Tail frac':>16}"
    f"{'Emax/T':>12}"
    f"{'Estado':>18}"
)

print(
    "-" * 115
)


convergence_results = []


# ============================================================
# FIGURA 1: Q_N VS N
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 7)
)

colors = plt.cm.viridis(
    np.linspace(
        0,
        1,
        len(T_CONVERGENCE)
    )
)


for T, color in zip(
    T_CONVERGENCE,
    colors
):

    Tval = T.to_value(
        u.K
    )

    Qcum = Q_acumulada(
        T
    )

    Qtot = Qcum[-1]

    N = np.arange(
        1,
        Nlevels + 1
    )


    # --------------------------------------------------------
    # Diferencia respecto a todos los niveles del archivo
    # --------------------------------------------------------

    remaining = (
        Qtot - Qcum
    ) / Qtot


    indices = np.where(
        remaining
        <= CONVERGENCE_TOL
    )[0]


    if len(indices) > 0:

        Nconv = (
            indices[0] + 1
        )

    else:

        Nconv = Nlevels


    # --------------------------------------------------------
    # Contribución del último 10 %
    # --------------------------------------------------------

    Ntail = max(
        1,
        int(
            Nlevels
            * TAIL_FRACTION_LEVELS
        )
    )

    i_tail = (
        Nlevels - Ntail
    )


    if i_tail > 0:

        tail_fraction = (
            Qcum[-1]
            - Qcum[i_tail - 1]
        ) / Qcum[-1]

    else:

        tail_fraction = 1.0


    # --------------------------------------------------------
    # Emax / T
    # --------------------------------------------------------

    Emax_over_T = (
        Evals[-1]
        / Tval
    )


    # --------------------------------------------------------
    # Estado
    # --------------------------------------------------------

    if (
        tail_fraction
        < CONVERGENCE_TOL
    ):

        estado = "CONVERGE"

    else:

        estado = "NO CONVERGE"


    convergence_results.append([
        Tval,
        Qtot,
        Nconv,
        Nconv / Nlevels,
        tail_fraction,
        Emax_over_T
    ])


    print(
        f"{Tval:12.2f}"
        f"{Qtot:18.6e}"
        f"{Nconv:12d}"
        f"{Nconv/Nlevels:12.4f}"
        f"{tail_fraction:16.6e}"
        f"{Emax_over_T:12.4f}"
        f"{estado:>18}"
    )


    ax.plot(
        N,
        Qcum,
        color=color,
        label=f"{Tval:.1f} K"
    )


ax.set_xlabel(
    "Número de niveles incluidos, N"
)

ax.set_ylabel(
    r"$Q_N(T)$"
)

ax.set_yscale(
    "log"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    fontsize=7,
    ncol=2
)

plt.tight_layout()

plt.savefig(
    "01_Q_convergencia_Nlevels.png",
    dpi=250
)

plt.close()


# ============================================================
# FIGURA 2: ERROR RELATIVO DE Q_N
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 7)
)


for T, color in zip(
    T_CONVERGENCE,
    colors
):

    Tval = T.to_value(
        u.K
    )

    Qcum = Q_acumulada(
        T
    )

    Qtot = Qcum[-1]

    remaining = (
        Qtot - Qcum
    ) / Qtot

    remaining = np.maximum(
        remaining,
        1.0e-16
    )

    ax.plot(
        np.arange(
            1,
            Nlevels + 1
        ),
        remaining,
        color=color,
        label=f"{Tval:.1f} K"
    )


ax.axhline(
    CONVERGENCE_TOL,
    color="black",
    linestyle="--",
    label=(
        f"Tolerancia = "
        f"{CONVERGENCE_TOL:.0e}"
    )
)

ax.set_xlabel(
    "Número de niveles incluidos, N"
)

ax.set_ylabel(
    r"$(Q_{\rm total}-Q_N)/Q_{\rm total}$"
)

ax.set_yscale(
    "log"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    fontsize=7,
    ncol=2
)

plt.tight_layout()

plt.savefig(
    "02_Q_convergencia_error.png",
    dpi=250
)

plt.close()


# ============================================================
# FIGURA 3: NÚMERO ACUMULADO DE ESTADOS
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.loglog(
    E_sorted,
    Gcum,
    label="Estados acumulados"
)


# ------------------------------------------------------------
# Región utilizada para el ajuste
# ------------------------------------------------------------

Efit = E_sorted[
    mask_fit_states
]


# ------------------------------------------------------------
# Modelo ajustado
# ------------------------------------------------------------

E_model = np.logspace(
    np.log10(
        Efit.min()
    ),
    np.log10(
        E_sorted.max()
    ),
    500
)

G_model = (
    A_states
    * E_model**beta_states
)


ax.loglog(
    E_model,
    G_model,
    "--",
    linewidth=2,
    label=(
        rf"$G(E)\propto "
        rf"E^{{{beta_states:.2f}}}$"
    )
)


ax.axvspan(
    Efit.min(),
    Efit.max(),
    alpha=0.15,
    label="Región de ajuste"
)


ax.axvline(
    Evals.max(),
    linestyle=":",
    label=r"$E_{\rm max}$"
)


ax.set_xlabel(
    r"$E/k_B$ [K]"
)

ax.set_ylabel(
    r"$G(E)=\sum_{E_i<E} g_i$"
)

ax.grid(
    True,
    which="both",
    alpha=0.3
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "03_cumulative_density_of_states.png",
    dpi=250
)

plt.close()


# ============================================================
# CALCULAR Q(T) CON Y SIN EXTRAPOLACIÓN
# ============================================================

Q_known_array = []
Q_missing_array = []
Q_estimated_array = []
missing_fraction_array = []


for T in T_Q:

    qk, qm, qe = estimate_Q_complete(
        T,
        Evals,
        g,
        C_rho,
        alpha_rho
    )

    Q_known_array.append(
        qk
    )

    Q_missing_array.append(
        qm
    )

    Q_estimated_array.append(
        qe
    )


    if (
        np.isfinite(qe)
        and qe > 0
    ):

        missing_fraction_array.append(
            qm / qe
        )

    else:

        missing_fraction_array.append(
            np.nan
        )


Q_known_array = np.array(
    Q_known_array
)

Q_missing_array = np.array(
    Q_missing_array
)

Q_estimated_array = np.array(
    Q_estimated_array
)

missing_fraction_array = np.array(
    missing_fraction_array
)


# ============================================================
# FIGURA 4: Q(T)
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)


ax.loglog(
    T_Q.value,
    Q_known_array,
    label="Niveles explícitos"
)


ax.loglog(
    T_Q.value,
    Q_estimated_array,
    "--",
    linewidth=2,
    label="Q con extrapolación"
)


ax.scatter(
    temps_table,
    Qvals_table,
    s=45,
    label="Valores tabulados",
    zorder=5
)


ax.set_xlabel(
    "Temperatura [K]"
)

ax.set_ylabel(
    r"$Q(T)$"
)

ax.grid(
    True,
    which="both",
    alpha=0.3
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "04_Q_known_vs_extrapolated.png",
    dpi=250
)

plt.close()


# ============================================================
# FIGURA 5: FRACCIÓN DE Q ESTIMADA COMO FALTANTE
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)


valid = (
    np.isfinite(
        missing_fraction_array
    )
    &
    (
        missing_fraction_array
        > 0
    )
)


ax.loglog(
    T_Q.value[valid],
    missing_fraction_array[valid]
)


ax.axhline(
    1.0e-3,
    linestyle="--",
    label="0.1 %"
)

ax.axhline(
    1.0e-2,
    linestyle=":",
    label="1 %"
)

ax.axhline(
    1.0e-1,
    linestyle="-.",
    label="10 %"
)


ax.set_xlabel(
    "Temperatura [K]"
)

ax.set_ylabel(
    r"$Q_{\rm missing}/Q_{\rm estimated}$"
)

ax.grid(
    True,
    which="both",
    alpha=0.3
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "05_missing_fraction.png",
    dpi=250
)

plt.close()


# ============================================================
# COMPARACIÓN CONTRA VALORES TABULADOS
# ============================================================

print()
print("=" * 110)
print("COMPARACIÓN CON VALORES TABULADOS")
print("=" * 110)

print(
    f"{'T [K]':>10}"
    f"{'Q tabla':>18}"
    f"{'Q known':>18}"
    f"{'Q estimado':>18}"
    f"{'err known':>18}"
    f"{'err estimado':>18}"
)

print(
    "-" * 110
)


comparison_results = []


for Ttab, Qtab in zip(
    temps_table,
    Qvals_table
):

    qk, qm, qe = estimate_Q_complete(
        Ttab * u.K,
        Evals,
        g,
        C_rho,
        alpha_rho
    )


    err_known = (
        qk - Qtab
    ) / Qtab


    if np.isfinite(qe):

        err_est = (
            qe - Qtab
        ) / Qtab

    else:

        err_est = np.nan


    print(
        f"{Ttab:10.3f}"
        f"{Qtab:18.6e}"
        f"{qk:18.6e}"
        f"{qe:18.6e}"
        f"{err_known:18.4%}"
        f"{err_est:18.4%}"
    )


    comparison_results.append([
        Ttab,
        Qtab,
        qk,
        qm,
        qe,
        err_known,
        err_est
    ])


comparison_results = np.array(
    comparison_results
)


# ============================================================
# TEMPERATURAS ESPECÍFICAS DE INTERÉS
# ============================================================

check_temperatures = np.array([
    10,
    20,
    50,
    100,
    200,
    300,
    500,
    1000,
    2000,
    3000,
    5000,
    10000,
    20000,
    50000,
    100000
])


print()
print("=" * 110)
print("ESTIMACIÓN DE NIVELES FALTANTES")
print("=" * 110)

print(
    f"{'T [K]':>10}"
    f"{'Q known':>18}"
    f"{'Q missing':>18}"
    f"{'Q estimated':>18}"
    f"{'Missing [%]':>16}"
    f"{'Emax/T':>14}"
)

print(
    "-" * 110
)


estimated_results = []


for Tval in check_temperatures:

    qk, qm, qe = estimate_Q_complete(
        Tval * u.K,
        Evals,
        g,
        C_rho,
        alpha_rho
    )


    if (
        np.isfinite(qe)
        and qe > 0
    ):

        frac = (
            qm / qe
        )

    else:

        frac = np.nan


    Emax_over_T = (
        Evals[-1]
        / Tval
    )


    print(
        f"{Tval:10.1f}"
        f"{qk:18.6e}"
        f"{qm:18.6e}"
        f"{qe:18.6e}"
        f"{100*frac:16.4f}"
        f"{Emax_over_T:14.4f}"
    )


    estimated_results.append([
        Tval,
        qk,
        qm,
        qe,
        frac,
        Emax_over_T
    ])


estimated_results = np.array(
    estimated_results
)


# ============================================================
# GUARDAR TABLA DE CONVERGENCIA
# ============================================================

convergence_results = np.array(
    convergence_results
)


np.savetxt(
    "Q_convergence_summary.txt",
    convergence_results,
    fmt="%.8e",
    header=(
        "T_K "
        "Q_known "
        "N_convergence "
        "Nconv_over_Ntotal "
        "tail_fraction "
        "Emax_over_T"
    )
)


# ============================================================
# GUARDAR COMPARACIÓN CON TABLA
# ============================================================

np.savetxt(
    "Q_table_comparison.txt",
    comparison_results,
    fmt="%.8e",
    header=(
        "T_K "
        "Q_table "
        "Q_known "
        "Q_missing "
        "Q_estimated "
        "relative_error_known "
        "relative_error_estimated"
    )
)


# ============================================================
# GUARDAR TEMPERATURAS DE INTERÉS
# ============================================================

np.savetxt(
    "Q_missing_estimates.txt",
    estimated_results,
    fmt="%.8e",
    header=(
        "T_K "
        "Q_known "
        "Q_missing "
        "Q_estimated "
        "missing_fraction "
        "Emax_over_T"
    )
)


# ============================================================
# GUARDAR TABLA COMPLETA DE Q(T)
# ============================================================

np.savetxt(
    "Q_extrapolated_full_table.txt",
    np.column_stack([
        T_Q.value,
        Q_known_array,
        Q_missing_array,
        Q_estimated_array,
        missing_fraction_array
    ]),
    fmt="%.8e",
    header=(
        "T_K "
        "Q_known "
        "Q_missing "
        "Q_estimated "
        "missing_fraction"
    )
)


# ============================================================
# GUARDAR Q EXTRAPOLADA EN FORMATO QUE ESTABAS UTILIZANDO
# ============================================================

np.savetxt(
    "../inputs/partitionfunction_ch3oh_extrapolated.inp",
    np.column_stack([
        T_Q.value,
        Q_estimated_array
    ]),
    fmt="%.8e",
    header=(
        f"1\n"
        f"{len(T_Q)}"
    ),
    comments=""
)


# ============================================================
# GUARDAR TAMBIÉN Q SIN EXTRAPOLACIÓN
# ============================================================

np.savetxt(
    "../inputs/partitionfunction_ch3oh_known.inp",
    np.column_stack([
        T_Q.value,
        Q_known_array
    ]),
    fmt="%.8e",
    header=(
        f"1\n"
        f"{len(T_Q)}"
    ),
    comments=""
)


# ============================================================
# RESUMEN FINAL
# ============================================================

print()
print("=" * 80)
print("RESUMEN FINAL")
print("=" * 80)

print(
    f"Número total de niveles : "
    f"{Nlevels}"
)

print(
    f"Emax/k                  : "
    f"{Evals[-1]:.4f} K"
)

print(
    f"A                       : "
    f"{A_states:.6e}"
)

print(
    f"beta                    : "
    f"{beta_states:.6f}"
)

print(
    f"alpha                   : "
    f"{alpha_rho:.6f}"
)

print(
    f"C                       : "
    f"{C_rho:.6e}"
)

print(
    f"Q asintótico            : "
    f"T^{beta_states:.4f}"
)

print(
    f"R^2 ajuste              : "
    f"{R2_states:.6f}"
)


print()
print(
    "Figuras generadas:"
)

print(
    "  01_Q_convergencia_Nlevels.png"
)

print(
    "  02_Q_convergencia_error.png"
)

print(
    "  03_cumulative_density_of_states.png"
)

print(
    "  04_Q_known_vs_extrapolated.png"
)

print(
    "  05_missing_fraction.png"
)


print()
print(
    "Tablas generadas:"
)

print(
    "  Q_convergence_summary.txt"
)

print(
    "  Q_table_comparison.txt"
)

print(
    "  Q_missing_estimates.txt"
)

print(
    "  Q_extrapolated_full_table.txt"
)


print()
print(
    "Archivos de función de partición:"
)

print(
    "  ../inputs/"
    "partitionfunction_ch3oh_known.inp"
)

print(
    "  ../inputs/"
    "partitionfunction_ch3oh_extrapolated.inp"
)


print()
print(
    "Terminado."
)