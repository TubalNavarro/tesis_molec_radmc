#######Script que ejecuta el main en paralelo###########

from yso_models import *
from make_line import *
from pathlib import Path
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from Line_max import analyze_fits_cube


# Número de procesos en paralelo
NPROC = 4


cwd = Path(__file__).resolve().parent


def organize_folder(modelname, molec="co"):
    """
    Crea la carpeta del modelo y copia los archivos necesarios.
    Cada proceso entra a su propia carpeta, por eso no se estorban.
    """

    os.chdir(cwd)

    model_dir = cwd / modelname
    model_dir.mkdir(exist_ok=True)

    shutil.copy2(
        cwd / "inputs" / "dustkappa_silicate.inp",
        model_dir / "dustkappa_silicate.inp"
    )

    shutil.copy2(
        cwd / "inputs" / f"molecule_{molec}.inp",
        model_dir / f"molecule_{molec}.inp"
    )

    shutil.copy2(
        cwd / "make_line.py",
        model_dir / "make_line.py"
    )

    shutil.copy2(
        cwd / "inputs" / f"partitionfunction_{molec}.inp",
        model_dir / f"partitionfunction_{molec}.inp"
    )

    os.chdir(model_dir)

    return model_dir


def run_one_model(task):
    """
    Ejecuta un solo modelo.
    Esta función es la que se manda a cada procesador.
    """

    modelname, ulrich_kwargs, line_kwargs = task

    try:
        print(f"[PID {os.getpid()}] Iniciando {modelname}", flush=True)

        model_dir = organize_folder(modelname, molec="ch3oh")

        UlrichDisk(**ulrich_kwargs)
        make_line_image_freq(**line_kwargs)

        print(f"[PID {os.getpid()}] Terminado {modelname}", flush=True)

        return modelname, True, None

    except Exception:
        err = traceback.format_exc()

        # Guarda el error dentro de la carpeta del modelo
        error_dir = cwd / modelname
        error_dir.mkdir(exist_ok=True)

        with open(error_dir / "ERROR.log", "w") as f:
            f.write(err)

        print(f"[PID {os.getpid()}] ERROR en {modelname}", flush=True)

        return modelname, False, err


def add_task(tasks, modelname, ulrich_kwargs=None, line_kwargs=None):
    if ulrich_kwargs is None:
        ulrich_kwargs = {}

    if line_kwargs is None:
        line_kwargs = {}

    tasks.append((modelname, ulrich_kwargs, line_kwargs))


def build_tasks():
    tasks = []

    # -------------------------
    # Modelos SIN disco
    # -------------------------

    for incl in [0, 5, 10, 20, 30, 40, 45, 50, 60, 70, 80, 85, 90]:
        add_task(
            tasks,
            f"model_Ulrich_nodisk_i={incl}",
            ulrich_kwargs={"discFlag": False},
            line_kwargs={"incl": incl}
        )

    for r in [200, 300, 400, 500, 600, 1000, 1500]:
        add_task(
            tasks,
            f"model_Ulrich_nodisk_rdisc={r}",
            ulrich_kwargs={"Rdisc": r, "discFlag": False}
        )

    for A in [1, 3, 5, 7, 9]:
        add_task(
            tasks,
            f"model_Ulrich_dens_nodisk_Arho0={A}",
            ulrich_kwargs={"Arho0": A, "discFlag": False}
        )

    for M in [9, 15, 18, 20, 22, 24, 27]:
        add_task(
            tasks,
            f"model_Ulrich_nodisk_MStar={M}",
            ulrich_kwargs={"MStar": M, "discFlag": False}
        )

    for Mdot in [5e-5, 8e-5, 1e-4, 3e-4, 6e-4, 8e-4, 2e-3]:
        add_task(
            tasks,
            f"model_Ulrich_nodisk_Mrate={Mdot}",
            ulrich_kwargs={"MRate": Mdot, "discFlag": False}
        )

    for B in [0.5, 1, 3, 5, 7, 10]:
        add_task(
            tasks,
            f"model_Ulrich_nodisk_BT={B}",
            ulrich_kwargs={"BT": B, "discFlag": False}
        )

    # for T10 in [2000, 2500, 3000, 3200, 3500, 4000]:
    #     add_task(
    #         tasks,
    #         f"model_Ulrich_nodisk_T10={T10}",
    #         ulrich_kwargs={"T10Env": T10, "discFlag": False}
    #     )

    for exp in [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]:
        add_task(
            tasks,
            f"model_Ulrich_nodisk_exp={exp}",
            ulrich_kwargs={"exp_disc": exp, "discFlag": False}
        )

    for X in [1e-6, 2.2e-6, 5e-6, 1e-5, 2.2e-5]:
        add_task(
            tasks,
            f"model_Ulrich_nodisk_abund={X}",
            ulrich_kwargs={"molec_abund": X, "discFlag": False}
        )

    # -------------------------
    # Modelos CON disco
    # -------------------------

    for incl in [0, 5, 10, 20, 30, 40, 45, 50, 60, 70, 80, 85, 90]:
        add_task(
            tasks,
            f"model_Ulrich_i={incl}",
            ulrich_kwargs={},
            line_kwargs={"incl": incl}
        )

    for r in [200, 300, 400, 500, 600, 1000, 1500]:
        add_task(
            tasks,
            f"model_Ulrich_rdisc={r}",
            ulrich_kwargs={"Rdisc": r}
        )

    for A in [1, 3, 5, 7, 9]:
        add_task(
            tasks,
            f"model_Ulrich_dens_Arho0={A}",
            ulrich_kwargs={"Arho0": A}
        )

    # for T10 in [2000, 2500, 3000, 3200, 3500, 4000]:
    #     add_task(
    #         tasks,
    #         f"model_Ulrich_T10={T10}",
    #         ulrich_kwargs={"T10Env": T10}
    #     )

    for M in [9, 15, 18, 20, 22, 24, 27]:
        add_task(
            tasks,
            f"model_Ulrich_MStar={M}",
            ulrich_kwargs={"MStar": M}
        )

    for Mdot in [5e-5, 8e-5, 1e-4, 3e-4, 6e-4, 8e-4, 2e-3]:
        add_task(
            tasks,
            f"model_Ulrich_Mrate={Mdot}",
            ulrich_kwargs={"MRate": Mdot}
        )

    for B in [0.5, 1, 3, 5, 7, 10]:
        add_task(
            tasks,
            f"model_Ulrich_BT={B}",
            ulrich_kwargs={"BT": B}
        )

    for exp in [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]:
        add_task(
            tasks,
            f"model_Ulrich_exp={exp}",
            ulrich_kwargs={"exp_disc": exp}
        )

    for X in [1e-6, 2.2e-6, 5e-6, 1e-5, 2.2e-5]:
        add_task(
            tasks,
            f"model_Ulrich_abund={X}",
            ulrich_kwargs={"molec_abund": X}
        )

    return tasks


if __name__ == "__main__":

    os.chdir(cwd)

    tasks = build_tasks()

    print(f"Modelos totales: {len(tasks)}")
    print(f"Corriendo con {NPROC} procesos en paralelo\n", flush=True)

    n_ok = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=NPROC) as executor:
        futures = {
            executor.submit(run_one_model, task): task[0]
            for task in tasks
        }

        for future in as_completed(futures):
            modelname = futures[future]

            try:
                name, ok, err = future.result()

                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    print(f"\nFalló: {name}")
                    print(f"Revisa: {cwd / name / 'ERROR.log'}\n")

            except Exception:
                n_fail += 1
                print(f"\nError inesperado en {modelname}")
                print(traceback.format_exc())

    print("\nResumen:")
    print(f"  Modelos terminados correctamente: {n_ok}")
    print(f"  Modelos con error:              {n_fail}")