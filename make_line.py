####This script takes a given sf3dmodels model, runs radmc3d and makes raw and convolved cubes 

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
from discminer.core import Cube
from radio_beam import Beam
from astropy.coordinates import SpectralCoord
from spectral_cube import SpectralCube
import sys
sys.path.append('/Users/migueljaquez/Estudiantes/Tubal/repo_tesis-main/radmc3d-2.0/python/radmc3dPy')
import radmc3dPy.image as image
from astropy.coordinates import SkyCoord
from extract_pv import *
import os
import shutil
import subprocess

import sys
#sys.path.append('./')
#import image_mod

from argparse import ArgumentParser

# make a line observations and output in frequency units as a real observation.
def make_line_image_freq(incl=70):
    parser = ArgumentParser(prog='Make line images from sf3d model', description='Make line images and readable fits cube')
    
    ### RADMC simulation options 

     # Cube rest. frec. 233.7722702480E9  

    parser.add_argument('-restfreq', '--restfreq', default=335.582017e9, type=float,  
                        help="Rest frequency of the cube. DEFAULTS to  223.72571493e9 Hz") # jaquez
    parser.add_argument('-linefreq', '--linefreq', default=335.582017e9, type=float,  
                        help="Line frequency of the cube. DEFAULTS to  223.795666 Hz") # jaquez
    parser.add_argument('-v_sys', '--v_sys', default=-43.5, type=float,
                        help='Systemic velocity of the source in km/s. DEFAULTS TO 0.'  )
    parser.add_argument('-nchan', '--nchan', default=56, type=int,
                        help="Number of velocity channels to be computed. DEFAULTS to 101.")
    parser.add_argument('-dv', '--dv', default=0.436180476698, type=float,   
                        help="Delta velocity channel, line images will range from -dv*nchan/2 to dv*nchan/2 centered in restfreq. DEFAULTS to 5.0 km/s.") #Jaquez
    parser.add_argument('-sizeau', '--sizeau', default=6847.5, type=float,
                        help="Maximum sky window extent in au. DEFAULTS to 1200 au.")
    parser.add_argument('-nx', '--nx', default=249, type=int, 
                        help="Number of pixels per spatial dimension. DEFAULTS to 264")
    parser.add_argument('-incl', '--incl', default=incl, type=float,
                        help="Disc inclination. DEFAULTS to 50.32 deg.")
    parser.add_argument('-dpc', '--dpc', default=2500, type=float,
                        help="Distance to source in pc. DEFAULTS to 100 pc.")
    parser.add_argument('-radmc3d', '--radmc3d', default=1, type=int,
                        help="Run radmc3d? DEFAULTS to 0.")

    #convolution image options
    parser.add_argument('-c', '--convolve', action="store_false", 
                        help="Convolve datacube with beam.")
    parser.add_argument('-bmaj', '--bmaj', default=0.073, type=float,
                        help="Beam major axis. DEFAULTS to 0.05 arcsec.")
    parser.add_argument('-bmin', '--bmin', default=0.055, type=float,
                        help="Beam minor axis. DEFAULTS to 0.05 arcsec.")
    parser.add_argument('-bpa', '--bpa', default=54.4, type=float,
                        help="Beam position angle. DEFAULTS to 0.0 deg.")
    parser.add_argument("--add_noise", action="store_true", help="add gaussian noise to the cube") #jaquez
    parser.add_argument("--noise_std", default=1e-3, type=float, help="noise standard deviation") #jaquez

    # add a wcs 

    parser.add_argument('-coord','--coord', default='15h57m59.799s -53d58m00.528s', type=str, 
                        help='Coordinates of the object in the sky.' ) 
    parser.add_argument('-fi', '--fileimage', default='image_line.out', type=str,
                        help="Output image file name. DEFAULTS to 'image_line.out'.")
    parser.add_argument('-fc', '--filecube', default='cube.fits', type=str,
                        help="Output fits file. DEFAULTS to 'cube.fits'.")
    parser.add_argument('-transition', '--transition', default=1, type=int, help="Number of line transition in molecule.inp file"  )
    args = parser.parse_args()

    #************************
    #PATHS AND WORKING FOLDER
    #************************
    radmc3d = '/home/tubal/repo_tesis/radmc3d-2.0/src/radmc3d'
    workdir = './'
    fileimage = 'image_line.out'

    sc_center = SpectralCoord(args.restfreq*u.Hz, doppler_rest=args.restfreq*u.Hz, doppler_convention='radio') # is zero for now #Jaquez usfull if we want to observe thinks with high velocities    
    line_center = SpectralCoord(args.linefreq*u.Hz, doppler_rest=args.restfreq*u.Hz, doppler_convention='radio') # is zero for now #Jaquez usfull if we want to observe thinks with high velocities 
    line_center = line_center.with_radial_velocity_shift((args.v_sys)*(u.km/u.s)) 
    
    dv_half = (args.dv*args.nchan)/2   
    nu_min = line_center.with_radial_velocity_shift((-dv_half)*(u.km/u.s)) 
    nu_max = line_center.with_radial_velocity_shift((dv_half)*(u.km/u.s))  
    
    print(f"Central frequency = {args.restfreq:.6f}")
    print(f'sc_center = {sc_center}')
    print(f'The cube frequency range is from {nu_min} to {nu_max}')
    
    #**************
    #RUN RADMC3D
    #**************
    if args.radmc3d: 

        subprocess.run('%s image iline  %d widthkms %.5f linenlam %d npix %d sizeau %.1f incl %.1f'%(radmc3d, args.transition ,dv_half, args.nchan, args.nx, args.sizeau, args.incl), shell=True) 
        shutil.move('image.out', workdir+args.fileimage) 

    im = image.readImage(workdir+fileimage) #shape: (nrows, ncols, nchan)
    im.writeFits(fname=workdir+"radmc_output"+fileimage.replace(".out",".fits"),dpc=args.dpc, coord=args.coord,) 
    
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
    hd_mod['RESTFRQ'] = args.restfreq
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
            major= args.bmaj * u.arcsec,
            minor=args.bmin * u.arcsec,
            pa=args.bpa * u.deg
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
    noise_file='../inputs/G328_noise.dat'
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


    center = SkyCoord(
        "15h57m59.799s",
        "-53d58m00.528s",
        frame="icrs"
    )

    make_pv_diagram(cube_file=output_cube,
        output_file='pv.fits',
        center_coord=center,
        pa_deg=90,
        length_arcsec=2.7,
        width_arcsec=0.0550,
        spacing_arcsec=0.0110,
        restfreq_GHz=args.linefreq*1e-9,
    )


    compute_residuals('../pv/pv_G328.fits', 'pv.fits', 'pv_residuals.fits', floor=6e-3)

