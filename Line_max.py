import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


def analyze_fits_cube(filename, threshold=1e-3, plot=True):
    
    def get_spec_axis(header):
        naxis = header["NAXIS"]
        for i in range(1, naxis + 1):
            ctype = header.get(f"CTYPE{i}", "").upper()
            if any(k in ctype for k in ["FREQ", "VELO", "VRAD"]):
                return naxis - i
        return None

    def pixels_per_beam(header):
        bmaj = header.get("BMAJ")
        bmin = header.get("BMIN")
        cdelt1 = abs(header.get("CDELT1"))
        cdelt2 = abs(header.get("CDELT2"))

        if None in (bmaj, bmin, cdelt1, cdelt2):
            return None

        omega_beam = np.pi * bmaj * bmin / (4.0 * np.log(2.0))
        omega_pix = cdelt1 * cdelt2

        return omega_beam / omega_pix

    # --- cargar ---
    with fits.open(filename) as hdul:
        cube = np.squeeze(hdul[0].data).astype(float)
        hdr = hdul[0].header

    print("Shape:", cube.shape)
    print("BUNIT:", hdr.get("BUNIT"))

    # --- eje espectral ---
    spec_axis = get_spec_axis(hdr)
    if spec_axis is None:
        raise ValueError("No se encontró eje espectral")

    spatial_axes = tuple(i for i in range(cube.ndim) if i != spec_axis)

    # --- espectros ---
    spec_peak = np.nanmax(cube, axis=spatial_axes)      # Jy/beam
    spec_mean = np.nanmean(cube, axis=spatial_axes)     # Jy/beam

    ppb = pixels_per_beam(hdr)
    if ppb is not None:
        spec_sum_jy = np.nansum(cube, axis=spatial_axes) / ppb  # Jy
    else:
        spec_sum_jy = None

    # --- resultados ---
    peak_value = np.nanmax(spec_peak)
    detected = peak_value > threshold

    print("\n--- RESULTADOS ---")
    print(f"Pico (Jy/beam): {peak_value:.3e}")
    print(f"Umbral: {threshold:.3e}")
    print("¿Detectada?:", detected)

    if spec_sum_jy is not None:
        print(f"Pico integrado (Jy): {np.nanmax(spec_sum_jy):.3e}")
        print(f"Pixeles por beam: {ppb:.2f}")

    # --- graficas ---
    if plot:
        plt.figure(figsize=(8,5))
        plt.plot(spec_peak, label="Peak (Jy/beam)")
        plt.plot(spec_mean, label="Mean (Jy/beam)", alpha=0.7)
        plt.axhline(threshold, ls="--", label="Threshold")
        plt.xlabel("Canal")
        plt.ylabel("Jy/beam")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if spec_sum_jy is not None:
            plt.figure(figsize=(8,5))
            plt.plot(spec_sum_jy)
            plt.xlabel("Canal")
            plt.ylabel("Jy")
            plt.title("Flujo integrado")
            plt.tight_layout()
            plt.show()

    return {
        "spec_peak": spec_peak,
        "spec_mean": spec_mean,
        "spec_sum_jy": spec_sum_jy,
        "detected": detected
    }
    
