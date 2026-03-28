import numpy as np
import matplotlib.pyplot as plt

partition=np.loadtxt("../inputs/partitionfunction_ch3oh-a.inp", skiprows=2)
partition= np.array(partition)
plt.figure()
plt.plot(partition[:,0], partition[:,1])
temps=np.array([2.725, 5, 9.375, 18.75, 37.5, 75, 150, 225, 300, 500, 1000])
Qvals=10**np.array([1.0752,1.4268, 1.8931, 2.4393, 2.9642, 3.4660, 3.9890, 4.3220, 4.5685, 5.0092 , 5.5413 ])
plt.scatter(temps, Qvals)
plt.grid('true')
plt.yscale('log')
plt.show()
plt.close()

import pandas as pd


# Ruta del archivo
archivo = "Partition_Functions.txt"


filas = []

with open(archivo, "r") as f:
    for linea in f:
        linea = linea.strip()

        # Saltar líneas vacías
        if not linea:
            continue

        # Saltar encabezado y separador
        if linea.startswith("tag") or linea.startswith("="):
            continue

        partes = linea.split()

        # Verificar que haya suficientes columnas
        if len(partes) < 13:
            print(f"Línea ignorada por formato inesperado: {linea}")
            continue

        tag = partes[0]
        datos = partes[-11:]
        molecule = " ".join(partes[1:-11])

        filas.append([tag, molecule] + datos)

columnas = ["tag", "molecule"] + [f"Q{i}" for i in range(11)]
df = pd.DataFrame(filas, columns=columnas)

# Filtrar donde molecule empieza con "a-"
df_filtrado = df[df["molecule"].str.contains("CH3OH")]

print(df_filtrado)