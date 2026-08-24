from config import load_config


config = load_config("configs/G328.json")

print("Source:", config["source"]["name"])
print("Distance:", config["source"]["distance_pc"], "pc")
print("Molecule:", config["line"]["molecule"])

print(
    "Beam:",
    config["beam"]["major_arcsec"],
    "x",
    config["beam"]["minor_arcsec"],
    "arcsec"
)

print(
    "Grid:",
    config["physical_grid"]["npoints"]
)