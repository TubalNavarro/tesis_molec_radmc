from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np

pv_file = "../pv/pv_G328_flipped.fits"

with fits.open(pv_file) as hdul:
    data = np.squeeze(hdul[0].data)
    header = hdul[0].header

wcs = WCS(header)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection=wcs)

im = ax.imshow(data, origin="lower", aspect="auto", interpolation="nearest")
plt.colorbar(im, ax=ax)

ax.set_xlabel("Offset")
ax.set_ylabel("Spectral axis")
plt.tight_layout()
plt.savefig('pv_G328.png')