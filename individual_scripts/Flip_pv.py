from astropy.io import fits
import numpy as np

# ============================================================
# Archivos
# ============================================================
input_fits = "../pv/pv_G328.fits"
output_fits = "../pv/pv_G328_flipped_xy_carta_freq.fits"

# ============================================================
# Constantes
# ============================================================
c_ms = 299792458.0  # velocidad de la luz en m/s

# ============================================================
# Leer FITS original
# ============================================================
with fits.open(input_fits) as hdul:
    data = np.squeeze(hdul[0].data)
    header_old = hdul[0].header.copy()

if data.ndim != 2:
    raise ValueError(f"Este script espera un FITS 2D. data.ndim = {data.ndim}")

ny, nx = data.shape

print(f"Dimensiones: ny={ny}, nx={nx}")

# ============================================================
# Invertir datos en ambos ejes
# ============================================================
# Eje 0: vertical / espectral
# Eje 1: horizontal / offset
data_new = data[::-1, ::-1]

# ============================================================
# Leer información original
# ============================================================
restfreq = header_old["RESTFRQ"]

crpix2_old = header_old["CRPIX2"]
crval2_old = header_old["CRVAL2"]
cdelt2_old = header_old["CDELT2"]
ctype2_old = header_old["CTYPE2"].strip().upper()
cunit2_old = header_old["CUNIT2"].strip()

# ============================================================
# Funciones auxiliares
# ============================================================
def old_spectral_value(p_old):
    """
    Coordenada espectral original en pixel FITS p_old.
    p_old se cuenta desde 1.
    """
    return crval2_old + (p_old - crpix2_old) * cdelt2_old


def vrad_to_freq(v_ms, restfreq_hz):
    """
    Conversión radio: v = c (1 - nu / nu0)
    por tanto nu = nu0 (1 - v/c)
    """
    return restfreq_hz * (1.0 - v_ms / c_ms)


def freq_to_vrad(freq_hz, restfreq_hz):
    """
    Conversión radio inversa.
    """
    return c_ms * (1.0 - freq_hz / restfreq_hz)


# ============================================================
# Construir eje espectral nuevo en frecuencia
# ============================================================
# Después de invertir verticalmente:
# nuevo pixel p_new = 1 corresponde al viejo p_old = ny
# nuevo pixel p_new = ny corresponde al viejo p_old = 1

if ctype2_old.startswith("VRAD"):
    # El FITS original está en velocidad radial
    v_bottom_old = old_spectral_value(1)
    v_top_old = old_spectral_value(ny)

    # Después del flip vertical:
    # fila 1 nueva contiene la antigua fila superior
    v1_new = v_top_old
    vny_new = v_bottom_old

    # La velocidad nueva debe aumentar con el pixel, porque invertimos
    cdelt_v_new = (vny_new - v1_new) / (ny - 1)

    # Convertimos el eje lineal de velocidad a frecuencia
    # Si v aumenta, la frecuencia disminuye
    cdelt2_new_freq = -restfreq * cdelt_v_new / c_ms

    # Usamos CRVAL2 = RESTFRQ, como en el header que sí lee CARTA.
    # Entonces CRPIX2 es el pixel donde v = 0, es decir nu = RESTFRQ.
    crval2_new_freq = restfreq
    crpix2_new_freq = 1.0 - (vrad_to_freq(v1_new, restfreq) - restfreq) / cdelt2_new_freq

elif ctype2_old.startswith("FREQ"):
    # Si el FITS original ya estuviera en frecuencia
    f_bottom_old = old_spectral_value(1)
    f_top_old = old_spectral_value(ny)

    # Después del flip vertical
    f1_new = f_top_old
    fny_new = f_bottom_old

    cdelt2_new_freq = (fny_new - f1_new) / (ny - 1)

    # También forzamos CRVAL2 = RESTFRQ como en el header de referencia
    crval2_new_freq = restfreq
    crpix2_new_freq = 1.0 - (f1_new - crval2_new_freq) / cdelt2_new_freq

else:
    raise ValueError(f"No sé convertir CTYPE2={ctype2_old}. Esperaba VRAD o FREQ.")

# ============================================================
# Construir header limpio estilo CARTA
# ============================================================
header = fits.Header()

header["SIMPLE"] = (True, "conforms to FITS standard")
header["BITPIX"] = (-64, "array data type")
header["NAXIS"] = (2, "number of array dimensions")
header["NAXIS1"] = (nx, "length of axis 1")
header["NAXIS2"] = (ny, "length of axis 2")
header["WCSAXES"] = (2, "Number of coordinate axes")

# ============================================================
# Eje horizontal: OFFSET en arcsec, centrado en cero
# ============================================================
# El header que sí lee CARTA usa:
# CTYPE1 = OFFSET
# CUNIT1 = arcsec
# CDELT1 = 0.011
#
# Tu pixel scale original:
# CDELT1 = 3.055555555556e-06 deg = 0.011 arcsec

pixscale_arcsec = abs(header_old["CDELT1"]) * 3600.0

crpix1_new = (nx + 1) / 2.0
crval1_new = 0.0

# Como pediste invertir los datos horizontales, aquí hay dos opciones.
#
# Opción A: eje visual usual en CARTA:
# izquierda = offset negativo, derecha = offset positivo
cdelt1_new = +pixscale_arcsec
#
# Opción B: header estrictamente consistente con el flip horizontal:
# izquierda = offset positivo, derecha = offset negativo
# cdelt1_new = -pixscale_arcsec

header["CRPIX1"] = (crpix1_new, "Pixel coordinate of reference point")
header["CRPIX2"] = (crpix2_new_freq, "Pixel coordinate of reference point")

header["CDELT1"] = (cdelt1_new, "[arcsec] Coordinate increment at reference point")
header["CDELT2"] = (cdelt2_new_freq, "[Hz] Coordinate increment at reference point")

header["CUNIT1"] = ("arcsec", "Units of coordinate increment and value")
header["CUNIT2"] = ("Hz", "Units of coordinate increment and value")

header["CTYPE1"] = ("OFFSET", "Coordinate type code")
header["CTYPE2"] = ("FREQ", "Frequency (linear)")

header["CRVAL1"] = (crval1_new, "[arcsec] Coordinate value at reference point")
header["CRVAL2"] = (crval2_new_freq, "[Hz] Coordinate value at reference point")

# ============================================================
# Keywords que aparecen en el header que sí lee CARTA
# ============================================================
if "LONPOLE" in header_old:
    header["LONPOLE"] = header_old["LONPOLE"]
else:
    header["LONPOLE"] = (180.0, "[deg] Native longitude of celestial pole")

if "LATPOLE" in header_old:
    header["LATPOLE"] = header_old["LATPOLE"]

header["RESTFRQ"] = (restfreq, "[Hz] Line rest frequency")

if "MJDREF" in header_old:
    header["MJDREF"] = header_old["MJDREF"]
else:
    header["MJDREF"] = (0.0, "[d] MJD of fiducial time")

if "SPECSYS" in header_old:
    header["SPECSYS"] = header_old["SPECSYS"]
else:
    header["SPECSYS"] = ("LSRK", "Reference frame of spectral coordinates")

# Copiar otras keywords útiles si existen
for key in [
    "DATE-OBS",
    "MJD-OBS",
    "OBSGEO-X",
    "OBSGEO-Y",
    "OBSGEO-Z",
    "BMAJ",
    "BMIN",
    "BPA",
]:
    if key in header_old:
        header[key] = header_old[key]

# ============================================================
# Historial
# ============================================================
header.add_history("Data flipped along spectral and horizontal axes.")
header.add_history("Horizontal axis centered at zero and written in arcsec.")
header.add_history("Spectral axis rewritten as FREQ in Hz for CARTA compatibility.")
header.add_history("CRVAL2 set to RESTFRQ and CRPIX2 adjusted consistently.")

# ============================================================
# Guardar FITS
# ============================================================
fits.writeto(output_fits, data_new.astype(np.float64), header, overwrite=True)

print(f"\nGuardado: {output_fits}")

# ============================================================
# Verificación
# ============================================================
def coord(hdr, axis, p):
    return hdr[f"CRVAL{axis}"] + (p - hdr[f"CRPIX{axis}"]) * hdr[f"CDELT{axis}"]

x1 = coord(header, 1, 1)
x2 = coord(header, 1, nx)

f1 = coord(header, 2, 1)
f2 = coord(header, 2, ny)

v1 = freq_to_vrad(f1, restfreq)
v2 = freq_to_vrad(f2, restfreq)

print("\nHeader horizontal nuevo:")
print(f"  CTYPE1 = {header['CTYPE1']}")
print(f"  CUNIT1 = {header['CUNIT1']}")
print(f"  CRPIX1 = {header['CRPIX1']}")
print(f"  CRVAL1 = {header['CRVAL1']} arcsec")
print(f"  CDELT1 = {header['CDELT1']} arcsec/pixel")
print(f"  columna 1   = {x1:.6f} arcsec")
print(f"  columna {nx} = {x2:.6f} arcsec")

print("\nHeader espectral nuevo:")
print(f"  CTYPE2 = {header['CTYPE2']}")
print(f"  CUNIT2 = {header['CUNIT2']}")
print(f"  CRPIX2 = {header['CRPIX2']}")
print(f"  CRVAL2 = {header['CRVAL2']} Hz")
print(f"  CDELT2 = {header['CDELT2']} Hz/pixel")
print(f"  RESTFRQ = {header['RESTFRQ']} Hz")

print("\nRango espectral equivalente en velocidad radio:")
print(f"  fila 1   = {v1/1000:.6f} km/s")
print(f"  fila {ny} = {v2/1000:.6f} km/s")