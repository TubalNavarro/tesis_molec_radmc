import numpy as np
import matplotlib.pyplot as plt
filename = "molecule_ch3oh-a_parche.inp" 
file_out = "ch3oh-a_partition.inp" 
n= 248751 #For 2.5 to 500 K with step = 0.002 K 
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



T_vals = np.linspace(2.5,500, n)

x = np.arange(len(T_vals))
def Z(T_vals):
    T_vals = np.asarray(T_vals)              # (nT,)
    expo = -energies_cm1[:, None] * 1.43877688 / T_vals[None, :]  # (nE, nT)
    return np.sum(weights[:, None] * np.exp(expo), axis=0)         # (nT,)

y=Z(T_vals)

 
  


# Normalización
#arr_err= (np.full(len(x), Z_vals[counter])-arr_suma)/Z_vals[counter]

# Plot
plt.plot(T_vals, y)
   # plt.plot(x, arr_err, label=f"Err,T = {T} K")



plt.title("Función de partición como función de T")
plt.xlabel("T [K]")
plt.ylabel("Z(T)")
plt.grid(True)

#plt.savefig("Partition_extrapol.png", dpi=200)
plt.savefig("Z(T).png", dpi=200)
#plt.savefig("Partition_orig.png", dpi=200)

with open(file_out, "w") as f_out:
    f_out.write("1\n")
    f_out.write("%i\n"%len(T_vals))
    for i, T in enumerate(T_vals):
        f_out.write("%f     %f\n"%(T, y[i]))

plt.show()