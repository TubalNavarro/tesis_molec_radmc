#!/usr/bin/env python3
import re
import argparse

# Captura: LEVEL, ENERGY, WEIGHT, J_K
ROW_RE = re.compile(
    r"^\s*(?P<lvl>\d+)\s+(?P<E>[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s+"
    r"(?P<w>[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s+(?P<JK>-?\d+_-?\d+)\s*$"
)

def main():
    ap = argparse.ArgumentParser(description="Ordena niveles por energía y renumera LEVEL.")
    ap.add_argument("infile", help="Archivo de entrada con la tabla")
    ap.add_argument("-o", "--outfile", default=None, help="Archivo de salida")
    ap.add_argument("--keep-header", action="store_true",
                    help="Conserva líneas que empiezan con '!' al inicio del archivo de salida")
    args = ap.parse_args()

    out = args.outfile or (args.infile.rsplit(".", 1)[0] + "_sortedE.txt")

    header_lines = []
    rows = []

    with open(args.infile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.lstrip().startswith("!"):
                header_lines.append(line.rstrip("\n"))
                continue

            m = ROW_RE.match(line)
            if not m:
                # Si hay líneas vacías u otras cosas, simplemente se ignoran
                continue

            E = float(m.group("E"))
            w = float(m.group("w"))
            JK = m.group("JK")
            rows.append((E, w, JK))

    # Orden principal: energía ascendente.
    # Desempate opcional: por J_K para que sea estable (puedes quitarlo si no te interesa).
    rows.sort(key=lambda x: (x[0], x[2]))

    with open(out, "w", encoding="utf-8") as fo:
        if args.keep_header and header_lines:
            # imprime el/los headers originales
            for hl in header_lines:
                fo.write(hl + "\n")
        else:
            # header estándar si no quieres conservar el original
            fo.write("!LEVEL + ENERGY(CM-1) + WEIGHT + QUANTUM NOS.  J_K\n")

        for new_lvl, (E, w, JK) in enumerate(rows, start=1):
            fo.write(f"{new_lvl:4d}  {E:12.7f}  {w:8.1f}  {JK}\n")

    print(f"Listo: {len(rows)} filas ordenadas por energía -> {out}")

if __name__ == "__main__":
    main()
