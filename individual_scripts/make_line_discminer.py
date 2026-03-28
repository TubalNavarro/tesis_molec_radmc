import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
from discminer.core import Cube
from radio_beam import Beam
import radmc3dPy.image as image

import os
import shutil
import subprocess

import sys
#sys.path.append('./')
#import image_mod

from argparse import ArgumentParser

def make_line_image(trans=2):

    parser = ArgumentParser(prog='Make line images from sf3d model', description='Make line images and readable fits cube')
    parser.add_argument('-nchan', '--nchan', default=15, type=int,
                        help="Number of velocity channels to be computed. DEFAULTS to 101.")
    parser.add_argument('-vmax', '--vmax', default=9, type=float,
                        help="Maximum velocity channel, line images will range from -vmax to vmax. DEFAULTS to 5.0 km/s.")
    parser.add_argument('-nx', '--nx', default=264, type=int,
                        help="Number of pixels per spatial dimension. DEFAULTS to 264")
    parser.add_argument('-incl', '--incl', default=70.0, type=float,
                        help="Disc inclination. DEFAULTS to 50.32 deg.")
    parser.add_argument('-sizeau', '--sizeau', default=4000, type=float,
                        help="Maximum sky window extent in au. DEFAULTS to 1200 au.")
    parser.add_argument('-dpc', '--dpc', default=1200, type=float,
                        help="Distance to source in pc. DEFAULTS to 100 pc.")

    parser.add_argument('-bmaj', '--bmaj', default=0.109, type=float,
                        help="Beam major axis. DEFAULTS to 0.05 arcsec.")
    parser.add_argument('-bmin', '--bmin', default=0.067, type=float,
                        help="Beam minor axis. DEFAULTS to 0.05 arcsec.")
    parser.add_argument('-bpa', '--bpa', default=-42, type=float,
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
        subprocess.run('%s image iline  %d widthkms %.1f linenlam %d npix %d sizeau %.1f incl %.1f'%(radmc3d, trans ,args.vmax, args.nchan, args.nx, args.sizeau, args.incl), shell=True) #sizeau=2*xmax
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
        RESTFRQ = 233.795666e9, #/Rest Frequency (Hz) - Band 6
        SPECSYS = 'LSRK', #/Spectral reference frame 
        VELREF  = 257 #/1 LSR, 2 HEL, 3 OBS, +256 Radio
    )

    hdu = fits.PrimaryHDU()
    hdu.header.update(header)        
    print('####111111####')
    datacube = Cube(data, hdu.header, vchannels, args.dpc*u.pc, beam=beam, filename=workdir+args.filecube)
    print('####2222222####')
    if args.convolve:
        for i in range(args.nchan):
            datacube.data[i] = datacube.beam_area*convolve(datacube.data[i], datacube.beam_kernel, preserve_nan=False)

    datacube.writefits()


