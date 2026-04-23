import numpy as np
from astropy.io import fits


input_file = '/home/tubal/repo_tesis/cubes/DIHCA_cubes/shared_data/G335.78/G355.78+0.17_2_subcube.fits'
output_file = input_file.replace('.fits', 'fix_head.fits')

hdul = fits.open(input_file)


beam_table = hdul[1].data


colnames = {name.upper(): name for name in beam_table.names}

bmaj = np.array(beam_table[colnames['BMAJ']], dtype=float)
bmin = np.array(beam_table[colnames['BMIN']], dtype=float)
bpa  = np.array(beam_table[colnames['BPA']],  dtype=float)

valid = np.isfinite(bmaj) & np.isfinite(bmin) & np.isfinite(bpa)

bmaj_med = np.nanmedian(bmaj[valid])
bmin_med = np.nanmedian(bmin[valid])
bpa_med  = np.nanmedian(bpa[valid])

print('Median beam:')
print('BMAJ =', bmaj_med)
print('BMIN =', bmin_med)
print('BPA  =', bpa_med)


hdr = hdul[0].header
hdr['BMAJ'] = bmaj_med
hdr['BMIN'] = bmin_med
hdr['BPA']  = bpa_med


hdr['BUNIT'] = 'JY/BEAM'


hdul.writeto(output_file, overwrite=True)

hdul.close()

print('Archivo guardado en:', output_file)