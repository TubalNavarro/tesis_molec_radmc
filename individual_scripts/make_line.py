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

import os
import shutil
import subprocess

import sys
#sys.path.append('./')
#import image_mod

from argparse import ArgumentParser

# make a line observations and output in frequency units as a real observation.
def make_line_image_freq(trans=2):
    parser = ArgumentParser(prog='Make line images from sf3d model', description='Make line images and readable fits cube')
    
    ### RADMC simulation options #jaquez

     # Cube rest. frec. 233.7722702480E9  ##Object frec. +Delta v

    parser.add_argument('-restfreq', '--restfreq', default=233.7956660e9, type=float,  
                        help="Rest frequency of the transition to observe. DEFAULTS to  223.72571493e9 Hz") # jaquez
    parser.add_argument('-v_sys', '--v_sys', default=0, type=float,
                        help='Systemic velocity of the source in km/s. DEFAULTS TO 0.'  )
    parser.add_argument('-nchan', '--nchan', default=3, type=int,
                        help="Number of velocity channels to be computed. DEFAULTS to 101.")
    parser.add_argument('-dv', '--dv', default=0.574826553576917, type=float,   #jaquez
                        help="Delta velocity channel, line images will range from -dv*nchan/2 to dv*nchan/2 centered in restfreq. DEFAULTS to 5.0 km/s.") #Jaquez
    parser.add_argument('-nx', '--nx', default=264, type=int,
                        help="Number of pixels per spatial dimension. DEFAULTS to 264")
    parser.add_argument('-incl', '--incl', default=60, type=float,
                        help="Disc inclination. DEFAULTS to 50.32 deg.")
    parser.add_argument('-sizeau', '--sizeau', default=6000, type=float,
                        help="Maximum sky window extent in au. DEFAULTS to 1200 au.")
    parser.add_argument('-dpc', '--dpc', default=3200, type=float,
                        help="Distance to source in pc. DEFAULTS to 100 pc.")
    parser.add_argument('-radmc3d', '--radmc3d', default=1, type=int,
                        help="Run radmc3d? DEFAULTS to 0.")

    #convolution image options
    parser.add_argument('-c', '--convolve', action="store_false", #type=bool, #jaquez, we change this to a boolean
                        help="Convolve datacube with beam.")
    parser.add_argument('-bmaj', '--bmaj', default=0.109, type=float,
                        help="Beam major axis. DEFAULTS to 0.05 arcsec.")
    parser.add_argument('-bmin', '--bmin', default=0.067, type=float,
                        help="Beam minor axis. DEFAULTS to 0.05 arcsec.")
    parser.add_argument('-bpa', '--bpa', default=-42, type=float,
                        help="Beam position angle. DEFAULTS to 0.0 deg.")
    parser.add_argument("--add_noise", action="store_true", help="add gaussian noise to the cube") #jaquez
    parser.add_argument("--noise_std", default=1e-3, type=float, help="noise standard deviation") #jaquez

    # extra cube data: in construction
    # add a wcs 
    #Jaquez: add coord (cordinates) args to put the object in the sky. Usefull to compare match the models with the observations
    #parser.add_argument('-coord','--coord', default='03h10m05s -10d05m30s', type=str, #this will be an input in radmc cube image writing 
                        #help='Coordinates of the object in the sky. Usefull to open with casa' ) # Jaquez 
    parser.add_argument('-fi', '--fileimage', default='image_line.out', type=str,
                        help="Output image file name. DEFAULTS to 'image_line.out'.")
    parser.add_argument('-fc', '--filecube', default='cube.fits', type=str,
                        help="Output fits file. DEFAULTS to 'cube.fits'.")

    args = parser.parse_args()

    #************************
    #PATHS AND WORKING FOLDER
    #************************
    radmc3d = '/home/tubal/repo_tesis/radmc3d-2.0/src/radmc3d'
    workdir = './'
    #os.system('mkdir %s'%workdir)

    #**************
    #RUN RADMC3D
    #**************
    #chan0 = 1300.3602804464033 #-10 km/s
    #chan1 = 1300.4470340402677 #10 km/s
    #os.system('radmc3d image lambdarange %.3f %.3f nlam %d npix 200 sizeau 120 incl 45'%(chan0, chan1, nchan))
    # -----------------------------
    # central frequency in velocity #Jaquez
    # I add this to put the header in units of frequency instead of velocity
    sc_center = SpectralCoord(args.restfreq*u.Hz, doppler_rest=args.restfreq*u.Hz, doppler_convention='radio') # is zero for now #Jaquez usfull if we want to observe thinks with high velocities 
    #sc_center = SpectralCoord(args.v_sys*(u.km/u.s), doppler_rest=args.restfreq*u.Hz, doppler_convention='radio') # is zero for now #Jaquez usfull if we want to observe thinks with high velocities 

    
    # half velocity                #jaquez
    #dv_half = (args.dv*args.nchan)/2   #Jaquez
    # frequency band edges using radio convention   #jaquez
    ######
    #nu_min = sc_center.with_radial_velocity_shift(-dv_half*(u.km/u.s)) 
      #jaquez
    #nu_max = sc_center.with_radial_velocity_shift(+dv_half*(u.km/u.s))   # jaquez
    #dnu = ((nu_max - nu_min)/args.nchan).value

    print(f"Central frequency = {args.restfreq:.6f}")
    print(f'sc_center = {sc_center}')
    print(f'The cube frequency range is from {nu_min} to {nu_max}')

    if args.radmc3d:
        subprocess.run('%s image iline  %d widthkms %.5f linenlam %d npix %d sizeau %.1f incl %.1f vkms %f'%(radmc3d, trans ,args.dv*args.nchan, args.nchan, args.nx, args.sizeau, args.incl, args.v_sys), shell=True) #sizeau=2*xmax
        shutil.move('image.out', workdir+args.fileimage)
    dpix = np.arctan((args.sizeau*u.au / (args.nx-1)) / (args.dpc*u.pc).to(u.au)).to(u.deg)
    im = image.readImage(workdir+args.fileimage) #shape: (nrows, ncols, nchan)
    data = np.moveaxis(im.imageJyppix, -1, 0) #new shape: (nchan, nrows, ncols)
    data = np.moveaxis(data, -1, 1) #new shape: (nchan, ncols, nrows)
    ### header in frequency units 
    # revisar si el header que sale de la imagen anterior tiene las coordenadas 
    # va faltar pasar las unidades de los pixeles a grados para poder proyectarlo en el cielo, vamos a necesitar:
    # LONPOLE =   1.800000000000E+02                                                  
    # LATPOLE =   2.284163510774E+01  
    header_freq = dict(
        SIMPLE  = True, 
        BITPIX  = -32, 
        NAXIS   = 3,                                                  
        NAXIS1  = args.nx,
        NAXIS2  = args.nx,
        NAXIS3  = args.nchan,
        EXTEND  = True,                                                  
        BTYPE   = 'Intensity',                                                           
        OBJECT  = 'Sf3d-Radmc3d',                                                      
        BUNIT   = 'Jy/pixel',      
        CTYPE1  = 'RA---SIN',                                                            
        CRVAL1  = 245.0,        # revisar esto 
        CDELT1  = -dpix.value,
        CRPIX1  = int(0.5*args.nx)+1,
        CUNIT1  = 'deg',                                                            
        CTYPE2  = 'DEC--SIN',
        CRVAL2  = -30.0,        # revisar esto.
        CDELT2  = dpix.value,
        CRPIX2  = int(0.5*args.nx)+1,
        CUNIT2  = 'deg',
        CTYPE3  = 'FREQ', #/ Frequency 
        CRVAL3  = nu_min.value,
        CDELT3  = dnu,         #Jaquez
        CRPIX3  = 1,
        CUNIT3  = 'Hz',                                                            
        RESTFRQ = sc_center.value, #/Rest Frequency (Hz)  #Jaquez
        SPECSYS = 'LSRK', #/Spectral reference frame 
        VELREF  = 257 #/1 LSR, 2 HEL, 3 OBS, +256 Radio
    )

    # save the figure in Jy/pix with the header updated
    hdu = fits.PrimaryHDU(data=data)
    hdu.header.update(header_freq) 

    hdul = fits.HDUList([hdu])
    hdul.writeto(workdir+'raw_radmc_output_Jyppix_'+args.filecube, overwrite=True) 

    if args.convolve: #convolve the image
        synth_cube = SpectralCube.read(workdir+'raw_radmc_output_Jyppix_'+args.filecube)
        # set a beam with the same size as a pixel and a beam to the information
       
        point_beam = Beam(dpix.value * u.deg) #Beam(major=None, minor=None, pa=None, area=None, default_unit=Unit('arcsec'), meta=None
        cube_ = synth_cube.with_beam(point_beam)
        beam = Beam(
            major=args.bmaj*u.arcsec,
            minor=args.bmin*u.arcsec,
            pa=args.bpa*u.deg
        )
        

        if args.add_noise:
            conv_synth_cube = cube_.convolve_to(beam) + np.random.normal(0,scale=args.noise_std,size=cube_.shape)
        else:
            conv_synth_cube = cube_.convolve_to(beam)
        # save this new cube
        
        (conv_synth_cube.to(u.Jy/u.pixel)).write(workdir+'convolved_jypx'+args.filecube, overwrite=True)
        
        conv_synth_cube.to(u.Jy/u.beam).write(workdir+'convolved_jybeam'+args.filecube, overwrite=True)

        with fits.open(workdir+'convolved_jypx'+args.filecube, mode='update') as hdul:
            hdul[0].header['BUNIT'] = 'Jy/pixel' #Force Jy/pixel to be readable in CARTA
            hdul.flush()

# make a data cube in velocity units
def make_line_image_vel(trans=2):

    parser = ArgumentParser(prog='Make line images from sf3d model', description='Make line images and readable fits cube')
    parser.add_argument('-nchan', '--nchan', default=11, type=int,
                        help="Number of velocity channels to be computed. DEFAULTS to 101.")
    parser.add_argument('-vmax', '--vmax', default=5.0, type=float,
                        help="Maximum velocity channel, line images will range from -vmax to vmax. DEFAULTS to 5.0 km/s.")
    parser.add_argument('-nx', '--nx', default=264, type=int,
                        help="Number of pixels per spatial dimension. DEFAULTS to 264")
    parser.add_argument('-incl', '--incl', default=60, type=float,
                        help="Disc inclination. DEFAULTS to 50.32 deg.")
    parser.add_argument('-sizeau', '--sizeau', default=6000, type=float,
                        help="Maximum sky window extent in au. DEFAULTS to 1200 au.")
    parser.add_argument('-dpc', '--dpc', default=157.2, type=float,
                        help="Distance to source in pc. DEFAULTS to 100 pc.")

    parser.add_argument('-bmaj', '--bmaj', default=0.3, type=float,
                        help="Beam major axis. DEFAULTS to 0.05 arcsec.")
    parser.add_argument('-bmin', '--bmin', default=0.3, type=float,
                        help="Beam minor axis. DEFAULTS to 0.05 arcsec.")
    parser.add_argument('-bpa', '--bpa', default=0.0, type=float,
                        help="Beam position angle. DEFAULTS to 0.0 deg.")

    parser.add_argument('-radmc3d', '--radmc3d', default=1, type=int,
                        help="Run radmc3d? DEFAULTS to 0.")
    parser.add_argument('-c', '--convolve', default=1, type=int,
                        help="Convolve datacube with beam? DEFAULTS to 1.")

    parser.add_argument('-fi', '--fileimage', default='image_line.out', type=str,
                        help="Output image file name. DEFAULTS to 'image_line.out'.")
    parser.add_argument('-fc', '--filecube', default='cube.fits', type=str,
                        help="Output fits file. DEFAULTS to 'cube.fits'.")

    args = parser.parse_args()

    #************************
    #PATHS AND WORKING FOLDER
    #************************
    radmc3d = '/home/tubal/repo_tesis/radmc3d-2.0/src/radmc3d'
    workdir = './'
    #os.system('mkdir %s'%workdir)




    #**************
    #RUN RADMC3D
    #**************
    #chan0 = 1300.3602804464033 #-10 km/s
    #chan1 = 1300.4470340402677 #10 km/s
    #os.system('radmc3d image lambdarange %.3f %.3f nlam %d npix 200 sizeau 120 incl 45'%(chan0, chan1, nchan))

    if args.radmc3d:
        subprocess.run('%s image iline  %d widthkms %.1f linenlam %d npix %d sizeau %.1f incl %.1f vkms %f'%(radmc3d, trans ,args.vmax, args.nchan, args.nx, args.sizeau, args.incl), shell=True) #sizeau=2*xmax
        shutil.move('image.out', workdir+args.fileimage)

    #*******************************
    #READ IMAGE.OUT IN AND MAKE CUBE
    #*******************************
    im = image.readImage(workdir+args.fileimage) #shape: (nrows, ncols, nchan)
    data = np.moveaxis(im.imageJyppix, -1, 0) #new shape: (nchan, nrows, ncols)
    data = np.moveaxis(data, -1, 1) #new shape: (nchan, ncols, nrows)
    vchannels = np.linspace(-args.vmax, args.vmax, args.nchan)

    beam = Beam(
        major=args.bmaj*u.arcsec,
        minor=args.bmin*u.arcsec,
        pa=args.bpa*u.deg
    )

    dpix = np.arctan((args.sizeau*u.au / (args.nx-1)) / (args.dpc*u.pc).to(u.au)).to(u.deg)
    dchan = abs(vchannels[1]-vchannels[0])

    header = dict(
        SIMPLE  = True, 
        BITPIX  = -32, 
        NAXIS   = 3,                                                  
        NAXIS1  = args.nx,
        NAXIS2  = args.nx,
        NAXIS3  = args.nchan,
        EXTEND  = True,                                                  
        BMAJ    = beam.major.to(u.deg).value,
        BMIN    = beam.minor.to(u.deg).value,
        BPA     = beam.pa.to(u.deg).value,
        BTYPE   = 'Intensity',                                                           
        OBJECT  = 'Sf3d-Radmc3d',                                                      
        BUNIT   = 'Jy/beam',      
        CTYPE1  = 'RA---SIN',                                                            
        CRVAL1  = 245.0,
        CDELT1  = -dpix.value,
        CRPIX1  = int(0.5*args.nx)+1,
        CUNIT1  = 'deg',                                                            
        CTYPE2  = 'DEC--SIN',
        CRVAL2  = -30.0,
        CDELT2  = dpix.value,
        CRPIX2  = int(0.5*args.nx)+1,
        CUNIT2  = 'deg',
        CTYPE3  = 'VRAD', #/ Radio velocity (linear) 
        CRVAL3  = vchannels[0],
        CDELT3  = dchan,
        CRPIX3  = 1,
        CUNIT3  = 'km/s',                                                            
        RESTFRQ = 223.72571493e9, #/Rest Frequency (Hz) - Band 6
        SPECSYS = 'LSRK', #/Spectral reference frame 
        VELREF  = 257 #/1 LSR, 2 HEL, 3 OBS, +256 Radio
    )

    hdu = fits.PrimaryHDU()
    hdu.header.update(header)        

    datacube = Cube(data, hdu.header, vchannels, args.dpc*u.pc, beam=beam, filename=workdir+args.filecube)

    if args.convolve:
        for i in range(args.nchan):
            datacube.data[i] = datacube.beam_area*convolve(datacube.data[i], datacube.beam_kernel, preserve_nan=False)

    datacube.writefits()


