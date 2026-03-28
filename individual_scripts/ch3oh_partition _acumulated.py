import numpy as np
import matplotlib.pyplot as plt
filename = "molecule_ch3oh-a_parche.inp" 

#filename = "molecule_ch3oh-a_orig.inp"  

levels = []
energies_cm1 = []
weights = []
J = []
K = []

reading = False
found_start = False
found_end = False

with open(filename, "r") as f:
    for lineno, raw in enumerate(f, start=1):
        line = raw.strip()

        # Inicio de la tabla (más robusto que startswith)
        if ("!level" in line.lower()) and ("energy" in line.lower()):
            reading = True
            found_start = True
            # print(f"[DEBUG] Inicio tabla en línea {lineno}: {line}")
            continue

        # Fin de la tabla
        if "number of radiative transitions" in line.lower():
            found_end = True
            break

        if not reading:
            continue

        # Ignorar comentarios y líneas vacías
        if (not line) or line.startswith("!"):
            continue

        parts = line.split()
        if len(parts) < 4:
            # línea rara dentro de la tabla, la saltamos
            continue

        lvl = int(parts[0])
        E   = float(parts[1])
        g   = float(parts[2])
        qn  = parts[3]           # ej: "3_-1"

        j_str, k_str = qn.split("_")

        levels.append(lvl)
        energies_cm1.append(E)
        weights.append(g)
        J.append(int(j_str))
        K.append(int(k_str))

levels = np.array(levels, dtype=int)
energies_cm1 = np.array(energies_cm1, dtype=float)
weights = np.array(weights, dtype=float)
J = np.array(J, dtype=int)
K = np.array(K, dtype=int)

print("Encontró inicio de tabla:", found_start)
print("Encontró fin de tabla:", found_end)
print("Niveles leídos:", levels.size)

# Si quieres ver los primeros 5:
if levels.size > 0:
    for i in range(min(5, levels.size)):
        print(levels[i], energies_cm1[i], weights[i], f"{J[i]}_{K[i]}")

#arr_suma=np.zeros(258)
#for i_max in range(max(1, levels.size)):
#    suma=0
#    #print('i_max: %i'%i_max)
#    for i in range(i_max):
#        suma= weights[i]*np.exp(-energies_cm1[i]*1.4388/T)+suma
#      #  print('i: %i'%i)
#    arr_suma[i_max]=suma
#    print('suma: %f'%suma)
#arr_suma=arr_suma/arr_suma[257]
#
#print('partition function: ', arr_suma)
#fig= plt.plot(np.linspace(0,258,258), arr_suma)
#plt.savefig('Partition.png')

T_vals = [5,50, 100, 200, 300, 500]
nlevels = levels.size
x = np.arange(nlevels)

plt.figure()
counter=0
for T in T_vals:
    
    arr_suma = np.zeros(nlevels)

    for i_max in range(1, nlevels):
        suma = 0.0
        for i in range(i_max):
            suma += weights[i] * np.exp(-energies_cm1[i] * 1.4388 / T)

        arr_suma[i_max] = suma

    # Normalización
    #arr_err= (np.full(len(x), Z_vals[counter])-arr_suma)/Z_vals[counter]
    Z_max = arr_suma[-1]
    arr_suma /= arr_suma[-1]

    
    # Plot
    plt.plot(x, arr_suma, label=f"T = {T} K,  Z_max=%f"%Z_max)
   # plt.plot(x, arr_err, label=f"Err,T = {T} K")

    counter=counter+1
plt.title("Funciones de partición normalizadas, CDMS")
plt.xlabel("Nivel")
plt.ylabel("Z_i / Z_max")
plt.legend()
plt.grid(True)

#plt.savefig("Partition_extrapol.png", dpi=200)
plt.savefig("Partition_parche.png", dpi=200)
#plt.savefig("Partition_orig.png", dpi=200)
plt.show()