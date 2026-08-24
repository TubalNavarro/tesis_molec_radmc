####This script takes a given sf3dmodels model, runs radmc3d and makes raw and convolved cubes 

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
from discminer.core import Cube
from radio_beam import Beam
from astropy.coordinates import SpectralCoord
from spectral_cube import SpectralCube
from pathlib import Path as FilePath
import sys
sys.path.append('/Users/migueljaquez/Estudiantes/Tubal/repo_tesis-main/radmc3d-2.0/python/radmc3dPy')
import radmc3dPy.image as image
from astropy.coordinates import SkyCoord
from extract_pv import make_pv_diagram, compute_residuals
import os
import shutil
import subprocess

import sys



# make a line observations and output in frequency units as a real observation.
def make_line_image_freq(incl, config):

    # ==========================
    # CONFIGURATION LOAD
    # ==========================

    source_cfg = config["source"]
    line_cfg = config["line"]
    image_cfg = config["synthetic_image"]
    beam_cfg = config["beam"]
    pv_cfg = config["pv"]
    obs_cfg = config["observation"]

    # Source
    dpc = source_cfg["distance_pc"]
    ra = source_cfg["ra"]
    dec = source_cfg["dec"]
    coord = f"{ra} {dec}"
    v_sys = source_cfg["v_sys_kms"]

    # Molecular line
    restfreq = line_cfg["restfreq_hz"]
    linefreq = line_cfg["linefreq_hz"]
    transition = line_cfg["transition"]

    # Synthetic image
    nx = image_cfg["npix"]
    nchan = image_cfg["nchan"]
    dv = image_cfg["dv_kms"]
    pixel_scale = image_cfg["pixel_scale_arcsec"]

    # Beam
    bmaj = beam_cfg["major_arcsec"]
    bmin = beam_cfg["minor_arcsec"]
    bpa = beam_cfg["pa_deg"]

    # PV
    pv_pa = pv_cfg["pa_deg"]
    pv_length = pv_cfg["length_arcsec"]
    pv_width = pv_cfg["width_arcsec"]
    pv_spacing = pv_cfg["spacing_arcsec"]

    sizeau = nx * pixel_scale * dpc
    print(f"Synthetic image size = {sizeau:.2f} au")
    print(f"Pixel scale = {pixel_scale:.5f} arcsec/pixel")
    
    #************************
    #PATHS AND WORKING FOLDER
    #************************
    radmc3d = '/home/tubal/scisoft/radmc3d-2.0/src/radmc3d'
    workdir = './'
    fileimage = 'image_line.out'

    sc_center = SpectralCoord(restfreq*u.Hz, doppler_rest=restfreq*u.Hz, doppler_convention='radio') # is zero for now #Jaquez usfull if we want to observe thinks with high velocities    
    line_center = SpectralCoord(linefreq*u.Hz, doppler_rest=restfreq*u.Hz, doppler_convention='radio') # is zero for now #Jaquez usfull if we want to observe thinks with high velocities 
    line_center = line_center.with_radial_velocity_shift((v_sys)*(u.km/u.s)) 
    
    dv_half = (dv*nchan)/2   
    nu_min = line_center.with_radial_velocity_shift((-dv_half)*(u.km/u.s)) 
    nu_max = line_center.with_radial_velocity_shift((dv_half)*(u.km/u.s))  
    
    print(f"Central frequency = {restfreq:.6f}")
    print(f'sc_center = {sc_center}')
    print(f'The cube frequency range is from {nu_min} to {nu_max}')
    
    #**************
    #RUN RADMC3D
    #**************

    subprocess.run('%s image iline  %d widthkms %.5f linenlam %d npix %d sizeau %.1f incl %.1f'%(radmc3d, transition ,dv_half, nchan, nx, sizeau, incl), shell=True, check=True) 
    shutil.move('image.out', workdir+fileimage) 

    im = image.readImage(workdir+fileimage) #shape: (nrows, ncols, nchan)
    im.writeFits(fname=workdir+"radmc_output"+fileimage.replace(".out",".fits"),dpc=dpc, coord=coord,) 
    
    # a partir del header de salida de radmc vamos a generar el header 
    hdul = fits.open(workdir+"radmc_output"+fileimage.replace(".out",".fits")) 
    dat = hdul[0].data

    hd = hdul[0].header

    ## header del modelo 
    hd_mod = hd.copy()
    # freq (vel) keys
    hd_mod['CTYPE3'] = 'FREQ'
    hd_mod['SPECSYS'] = 'LSRK    '
    hd_mod['VELREF']  = 257 
    hd_mod['RESTFRQ'] = restfreq
    hd_mod['BUNIT'] = 'Jy/pixel'
    hd_mod['CUNIT3'] = 'Hz'
    hd_mod['CRVAL3'] = nu_min.value
    # sky position keys
    hd_mod['EQUINOX'] =  2.000000000000E+03
    hd_mod['RADESYS'] = 'FK5     ' 

    # crear el nuevo header
    hdu = fits.PrimaryHDU(data=dat,header=hd_mod)
    hdul = fits.HDUList([hdu])
    hdul.writeto(workdir+'raw_radmc_header_updated.fits', overwrite=True)
    

    syn_cube = SpectralCube.read(workdir+'raw_radmc_header_updated.fits') #Jy/pix
    hdr_syn = syn_cube.header

    #subtract continuum
    print('\n')
    print('Starting continuum_subtraction ')

    csub_image = 'raw_radmc_csub.fits'
    cont_image = 'raw_radmc_cont.fits'
    hdul = fits.open(workdir+'raw_radmc_header_updated.fits')
    data = hdul[0].data
    data2 = data.copy()
    datac = data.copy()	

    for x in range(data.shape[2]):
        for y in range(data.shape[1]):
            data2[:,y,x] = data[:,y,x] - data[:,y,x].min()
            datac[:,y,x] = data[:,y,x].min()
    
    fits.writeto(csub_image,data2,hdul[0].header,overwrite=True)
    fits.writeto(cont_image,datac,hdul[0].header,overwrite=True)	#QZ


    def convolve(cube_name):
    # define a beam of the size of the pixel
        clean_name = cube_name.replace("raw_radmc_", "").replace("header_updated", "")

        syn_cube = SpectralCube.read(workdir+cube_name+'.fits') #Jy/pix
        point_beam = Beam(0.0 * u.deg) 
        cube_ = syn_cube.with_beam(point_beam) # queda en unidades de Jy.pix-1
    
        new_beam = Beam(
            major= bmaj * u.arcsec,
            minor=bmin * u.arcsec,
            pa=bpa * u.deg
        )
    
        conv_synth_cube = cube_.convolve_to(new_beam)
        # save this new cube
        conv_synth_cube.write(workdir+clean_name+'_convolved_Jyppix.fits', overwrite=True)
        ### aqui tengo problemas para obtener un cubo con un header que casa lea correctamente


        conv_synth_cube.to(u.Jy/u.beam).write(workdir+cube_name+'_convolved_Jypbeam.fits', overwrite=True)
    
        hdul_ = fits.open(workdir+cube_name+'_convolved_Jypbeam.fits')
        data_ = hdul_['PRIMARY'].data
        header_ = hdul_['PRIMARY'].header
        new_header = conv_synth_cube.header.copy()
        # quito el key BEAM?
        del new_header['BEAM']
        new_header['BUNIT'] = 'JY/BEAM'
        
        #Final cube Jy/beam
        hdu_conv = fits.PrimaryHDU(data=data_,header=new_header)
        hdul_conv = fits.HDUList([hdu_conv])
        hdul_conv.writeto(workdir+clean_name+'_convolved_Jypbeam_header_update.fits', overwrite=True)

    convolve('raw_radmc_header_updated')
    convolve('raw_radmc_csub')
    convolve('raw_radmc_cont')





    cube_name='csub_convolved_Jypbeam_header_update.fits'
    noise_file = FilePath(obs_cfg["noise_file"]).name
    output_cube = "csub_convolved_noise.fits"
    random_seed = 42
    noise_per_channel = np.loadtxt(noise_file)
    noise_per_channel = np.atleast_1d(noise_per_channel)
    with fits.open(cube_name) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header.copy()

    nchan = data.shape[0]

    rng = np.random.default_rng(random_seed)
    data_noisy = data.copy()
    for i in range(nchan):
        sigma = noise_per_channel[i]
        noise = rng.normal(loc=0.0, scale=sigma, size=data[i, :, :].shape)
        data_noisy[i, :, :] += noise
    fits.writeto(output_cube, data_noisy, header, overwrite=True)


##########MAKE PV################


    center = SkyCoord(ra, dec, frame="icrs")

    make_pv_diagram(cube_file=output_cube,
        output_file='pv.fits',
        center_coord=center,
        pa_deg=pv_pa,
        length_arcsec=pv_length,
        width_arcsec=pv_width,
        spacing_arcsec=pv_spacing,
        restfreq_GHz=linefreq*1e-9,
    )

    observed_pv_file = FilePath(obs_cfg["pv_file"]).name
    compute_residuals(observed_pv_file, 'pv.fits','pv_residuals.fits')
        # Leer imagen del modelo teorico

    hdu_model = fits.open('pv.fits')
    model_data = hdu_model[0].data
    hdu_model.close()
    return model_data

    