import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom


def resample_pv(input_fits, output_fits, new_npix=123):

    with fits.open(input_fits) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()

    # Asumiendo:
    # data.shape = (velocidad, espacio)
    # FITS axis 1 = espacio
    # FITS axis 2 = velocidad

    old_npix = data.shape[1]

    # Factor de resampleo espacial
    zoom_factor = new_npix / old_npix

    # Eje velocidad se queda igual (factor 1)
    new_data = zoom(
        data,
        zoom=(1, zoom_factor),
        order=1
    )

    # Actualizar header espacial
    old_cdelt = header["CDELT1"]
    old_crpix = header["CRPIX1"]

    # Mantener el mismo rango espacial entre
    # el primer y último centro de píxel
    scale = (old_npix - 1) / (new_npix - 1)

    header["CDELT1"] = old_cdelt * scale
    header["CRPIX1"] = 1 + (old_crpix - 1) / scale
    header["NAXIS1"] = new_npix

    fits.writeto(
        output_fits,
        new_data,
        header,
        overwrite=True
    )

    print("Shape original:", data.shape)
    print("Shape nueva:   ", new_data.shape)
    print("CDELT1:", old_cdelt, "->", header["CDELT1"])
    print("CRPIX1:", old_crpix, "->", header["CRPIX1"])



# =============================================================
# USO
# =============================================================

resample_pv(
    input_fits="../pv/pv_G328_flipped_xy_freq.fits",
    output_fits="pv_G328_resampled_flipped.fits",
    new_npix=123
)