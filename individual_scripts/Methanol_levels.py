import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as ct
import numpy as np

def leer_tabla(archivo):
    filas = []

    with open(archivo, "r") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue

            if ":" not in linea:
                continue

            izquierda, derecha = linea.split(":")

            # Energía (columna 0)
            energia = float(izquierda.split()[0])

            # Valores de la derecha (ignorando índice)
            valores = [int(v) for v in derecha.split()]

            # Unir todo en una sola fila
            fila = [energia] + valores

            filas.append(fila)

    return np.array(filas)

data=leer_tabla('c032504.egy')
E=(data[:,0]/u.cm).to(1/u.m)*(ct.h*ct.c/ct.k_B)
print(E)
mask_u = (
    (data[:, 1] == 18) &
    (data[:, 2] == 3) &
    (data[:, 3] == 15) &
    (data[:, 4] == 0) 
)

u_level = data[mask_u]

mask_l = (
    (data[:, 1] == 17) &
    (data[:, 2] == 4) &
    (data[:, 3] == 14) &
    (data[:, 4] == 0) 
)

l_level = data[mask_l]
g=(data[:,1]*2+1)*4

def Q(T):
    return np.sum(g*np.exp(-(E/T)))

Q(10*u.K)

T=np.linspace(2.725, 5000, num=20000) 

print(T[15])
Z=np.zeros(len(T))

for i, temp in enumerate(T):
    Z[i]=Q(temp*u.K)


temps=np.array([2.725, 5, 9.375, 18.75, 37.5, 75, 150, 225, 300, 500, 1000])
Qvals=10**np.array([1.0752,1.4268, 1.8931, 2.4393, 2.9642, 3.4660, 3.9890, 4.3220, 4.5685, 5.0092 , 5.5413 ])



plt.figure()
plt.plot(T, Z, label='Q(T) Calculada')
plt.scatter(temps, Qvals, label='Tabla Q(T)', color='black')
plt.grid('true')
plt.yscale('log')
plt.legend()
plt.savefig('Q(T).png')
plt.close()

np.savetxt(
    "../inputs/partitionfunction_ch3oh.inp",
    np.column_stack((T, Z)),
    fmt="%.6e",          # notación científica
    header=f"1\n{len(T)}",
    comments=""
)