from astropy import units as u
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from pvextractor import extract_pv_slice, Path
from astropy.coordinates import SkyCoord

# read the data
data_file = '/home/tubal/repo_tesis/cubes/DIHCA_cubes/shared_data/G335.78/G355.78+0.17_2_subcubefix_head.fits'
cube = SpectralCube.read(data_file)

max_map = cube.max(axis=0).value # axis=0 means in the axis of frequency
img = plt.imshow(max_map, cmap='viridis', origin='lower') #norm = colors.LogNorm())# norm = colors.AsinhNorm())
cbar = plt.colorbar(img, label=r'$\mathrm{Jy \ beam^{-1}}$')
plt.title('Max moment')
plt.show()

# moment0 = \int I_nu d_v  #integrated value of the spectrum
moment_0 = cube.moment(order=0).value # axis=0 means in the axis of frequency
#print(moment_0)
img = plt.imshow(moment_0, cmap='viridis',origin='lower' ) #norm = colors.LogNorm())# norm = colors.AsinhNorm())
cbar = plt.colorbar(img, label=r'$\mathrm{Jy \ beam^{-1}} \ km \ s^{-1}$')
plt.title('Moment 0')
plt.show()

# moment_1 = \int (v I_nu d_v)/M0    # intensity weighted coordinate #v = velocidad

# we need to add a mask
RMS = 2e-3*u.Jy/u.beam
mask = cube > 5*RMS
CH3OH_freq = 2.33795666e+11*u.Hz
cube_mask = cube.with_mask(mask)

cube3 = cube_mask.with_spectral_unit(u.km/u.s,
                                    velocity_convention='optical',
                                    rest_value=CH3OH_freq)
#optional:
#subcube = cube3.spectral_slab(-60*u.km/u.s,-20*u.km/u.s)
#moment_1 = subcube.moment(order=1).value
moment_1 = cube3.moment(order=1).value # axis=0 means in the axis of frequency
img = plt.imshow(moment_1, cmap='jet', norm = colors.AsinhNorm(linear_width=1), origin='lower')
cbar = plt.colorbar(img, label=r'$\mathrm{km \ s^{-1}}$')
plt.title('Moment 1')
plt.show()


from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
coords = SkyCoord(
    ra=[
        "16h29m46.08527s",
        "16h29m46.17421s"
    ],
    dec=[
        "-48d15m50.1583s",
        "-48d15m49.7441s"
    ],
    frame="icrs"
)

path = Path(coords, width=0.011*u.arcsec)


# ----------------------------
# Extraer PV
# ----------------------------
pv = extract_pv_slice(cube, path, spacing=0.001*u.arcsec)

# Guardar en FITS
pv.writeto("pv_PA65.fits", overwrite=True)

print("PV guardado como pv_PA65.fits")
print("Shape del PV:", pv.data.shape)

# ----------------------------
# Graficar PV
# ----------------------------
ww = WCS(pv.header)

plt.figure()
ax = plt.subplot(111, projection=ww)

im = ax.imshow(
    pv.data,
    cmap='viridis',
    origin='lower',
    aspect='auto'
)

plt.colorbar(im, ax=ax, label=r'$\mathrm{Jy\ beam^{-1}}$')
ax.set_xlabel("Offset")
ax.set_ylabel(r"$v_\mathrm{LSR}$")
plt.title("PV diagram PA = 65 deg")
plt.show()

# ----------------------------
# Mostrar corte sobre moment 0
# ----------------------------
plt.figure()
ax = plt.subplot(111, projection=cube.wcs.celestial)

im = ax.imshow(
    moment_0,
    origin='lower',
    cmap='viridis'
)

plt.colorbar(im, ax=ax, label=r'$\mathrm{Jy\ beam^{-1}\ km\ s^{-1}}$')

path.show_on_axis(
    ax,
    spacing=1,
    color='red'
)

ax.set_xlabel("RA")
ax.set_ylabel("Dec")
plt.title("PV cut over Moment 0")
plt.show()
