'''
Script that creates possible models of molecular emission for RADMC-3D
'''

#------------------
#Import the package
#------------------
from sf3dmodels import Model, Plot_model
from sf3dmodels import Resolution as Res
import sf3dmodels.utils.units as u
import sf3dmodels.rt as rt
import sf3dmodels.utils.constants as ct

#-----------------
#Extra libraries
#-----------------
from matplotlib import colors
import numpy as np
import os
import sys
import json
import time

# -----------------
# User parameters
# -----------------
discFlag = True
envFlag = True
nmodel = 0
MStar = 200
MRate = 1e-4
Rdisc = 2842
Arho0 = 10
Renv = 10000

# Parameters that used to be function arguments
cavity_ang = 60
i = 75
exp_disc = 0.6
prop_only = False

# -----------------
# Main script
# -----------------
t0 = time.time()

print('\n')
print('Passed paremeters are:')
print('nmodel: {}'.format(nmodel))
print('MStar: {}'.format(MStar))
print('MRate: {}'.format(MRate))
print('Rdisc: {}'.format(Rdisc))
print('Arho0: {}'.format(Arho0))
print('Renv: {}'.format(Renv))

#------------------
MStar = MStar * u.MSun
LStar = u.LSun * (MStar/u.MSun)**4

#-------------------------------
#Parameters for the Pringle disc
#-------------------------------
MRate = MRate * u.MSun_yr
RStar = u.RSun * (MStar/u.MSun)**0.8
print('RStar:'.format(RStar))
TStar = u.TSun * ((LStar/u.LSun) / (RStar/u.RSun)**2)**0.25
Rd = Rdisc * u.au

print('RStar:', RStar/u.RSun, ', LStar:', LStar/u.LSun, ', TStar:', TStar)

#---------------
#GRID Definition
#---------------
#Cubic grid, each edge ranges [-size, size] au.

sizex = sizey = sizez = 10000 * u.au
Nx = Ny = Nz = 100  # Number of divisions for each axis
GRID = Model.grid([sizex, sizey, sizez], [Nx, Ny, Nz],
                  rt_code='radmc3d', include_zero=True)
NPoints = GRID.NPoints  # Final number of nodes in the grid

#--------
#DENSITY
#--------
Rho0 = Res.Rho0(MRate, Rd, MStar)
Arho = Arho0  # Disc-envelope density factor
Renv = Renv * u.au  # Envelope radius
Cavity = cavity_ang * np.pi/180  # Cavity opening angle

density = Model.density_Env_Disc(
    RStar, Rd, Rho0, Arho, GRID, exp_disc=exp_disc,
    discFlag=discFlag, envFlag=envFlag, rho_min_env=1.0,
    renv_max=Renv, ang_cavity=Cavity,
    average_around_Rd=np.median
)

#---------------------
# MODEL TEMPERATURE
#---------------------
# Ionized gas temperature
t_e = 1.0e2  # K
#temperature = Model.temperature_Constant(density, GRID, discTemp=t_e, envTemp=t_e, backTemp=2.725480)
temperature = Model.temperature_Constant(density, GRID, discTemp=t_e, backTemp=2.725480)

#---------------------
# Calculation of recombination rate
#---------------------
print('############################################')
print('The GRID cell lengths are: ({0},{1},{2}) meters'.format(GRID.step[0], GRID.step[1], GRID.step[2]))
dv = GRID.step[0] * GRID.step[1] * GRID.step[2]  # m^3
mp = 1.6726e-27  # kg
Msun = 1.989e30  # kg
mass = mp * np.sum(density.total) * dv  # [kg m ^-3]*[m^3] = kg
print('Total mass in model is {0} kg'.format(mass))
print('Total mass in model is {0} Msun'.format(mass/Msun))
alpha_t = 2.6e-19  # case B recombination coeff in m^3 s^-1
phi_ion = alpha_t * dv * np.sum(density.total**2)
print('The ionizing photon rate phi is {0:.3e} s^-1'.format(phi_ion))
print('The logarithmic ionizing photon rate phi is {0:.3f} s^-1'.format(np.log10(phi_ion)))
print('############################################')

#--------
#VELOCITY
#--------
vel = Model.velocity(RStar, MStar, Rd, density, GRID)

Model.PrintProperties(density, temperature, GRID, species='dens_ion')
Model.PrintProperties(density, temperature, GRID, species='dens_e')

#-------------------------
#ROTATION, VSYS, CENTERING
#-------------------------

#xc, yc, zc = [0.0,0.0,0.0]
#CENTER = [xc, yc, zc] #New center of the region in the global grid
#newProperties = Model.ChangeGeometry(GRID, center = CENTER,  vel = vel,
#                                     rot_dict = {'angles': [0*(np.pi/2)*(60./90)], 'axis': ['x'] })
#GRID.XYZ = newProperties.newXYZ #XYZ redefinition
#vel.x, vel.y, vel.z = newProperties.newVEL #vels redefinition

#rot_dict = {'angles': [(np.pi/2)*(135./90),(np.pi/2)*(60./90)], 'axis': ['y','x'] })

#**********************
#WRITE RADMC-3D FILES
#**********************
abundance = 5e-7 + np.zeros(GRID.NPoints) #Abundance of the molecule with respect to H2 
gtdratio = Model.gastodust(100., GRID.NPoints)
microturb = 100. + np.zeros(GRID.NPoints)

prop = {
    'dens_H2': density.total,
    'dens_dust': 2*ct.mH * density.total * 1/gtdratio,  # mass density # ct.mH -> u.amu
    'temp_dust': temperature.total,  # 30+np.zeros_like(density.total),
    #'temp_gas': temperature.total,
    'velocity': [vel.x, vel.y, vel.z],
    'gtdratio': gtdratio,
    'microturbulence': microturb,
    'abundance': abundance 
}

if prop_only:
    # If you want to stop here and just inspect objects, keep this flag True.
    # In a script, "return" is invalid at top-level, so we just exit.
    print("prop_only=True -> exiting before writing RADMC-3D files.")
    sys.exit(0)

radmc = rt.Radmc3d(GRID)
wavelength_intervals = [1e-1, 1e3, 1e4]  # [5e-3, 5e1, 1e4] ¿¿¿¿¿Different for another line??????
wavelength_divisions = [40, 40]

radmc.write_radmc3d_control(
    nphot=100000000, incl_dust=1, setthreads=8,
    incl_freefree=0, tgas_eq_tdust=1,
    modified_random_walk=1
)
radmc.write_amr_grid()
radmc.write_dust_density(prop['dens_dust'])  # Mass density
radmc.write_dust_temperature(prop['temp_dust'])  # Spherical radial plaw temperature produces artifacts near the disc outer radius in the line images
radmc.write_gas_velocity(prop['velocity'])
radmc.write_microturbulence(prop['microturbulence'])

radmc.write_stars(
    nstars=1, pos=[[0, 0, 0]], rstars=[u.RSun], mstars=[MStar],
    flux=[[-u.TSun]],  # flux --> if negative, radmc assumes the input number as the blackbody temperature of the star
    lam=wavelength_intervals, nxx=wavelength_divisions
)
radmc.write_wavelength_micron(lam=wavelength_intervals, nxx=wavelength_divisions)  # lam in microns, nxx divisions

#********************************
#Write molecule files for RADMC-3D
#********************************
nx, ny, nz = GRID.Nodes

with open('numberdens_ch3oh_a.inp', 'w+') as f:
    f.write('1\n')                 # Format number
    f.write('%d\n' % (nx*ny*nz))    # Nr of cells
    data = prop['abundance'] * density.total * (100**-3)  # To 1/cm3 units
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')

# Lines file
with open('lines.inp', 'w+') as f:
    f.write('2\n')
    f.write('1\n')
    f.write('ch3oh_a    leiden    0    0    0\n') 

#
# Dust opacity control file
#
with open('dustopac.inp', 'w+') as f:
    f.write('2               Format number of this file\n')
    f.write('1               Nr of dust species\n')
    f.write('============================================================================\n')
    f.write('1               Way in which this dust species is read\n')
    f.write('0               0=Thermal grain\n')
    f.write('silicate        Extension of name of dustkappa_***.inp file\n')
    f.write('----------------------------------------------------------------------------\n')



#
# Write the radmc3d.inp control file
#
#with open('radmc3d.inp', 'w+') as f:
#    f.write('nphot = %d\n'%(nphot))
#    f.write('scattering_mode_max = 1\n')
#    f.write('iranfreqmode = 1\n')

#-------
#TIMING
#-------
print('Ellapsed time for iterarion of create_model.py: %.3fs' % (time.time() - t0))
print('-------------------------------------------------\n-------------------------------------------------\n')




#-----------------------------------------------
#3D Points Distribution (weighting with density)
#-----------------------------------------------

tag = 'Main'
dens_plot = density.total / 1e6

weight = 10*Rho0
r = GRID.rRTP[0] / u.au #GRID.rRTP hosts [r, R, Theta, Phi] --> Polar GRID
Plot_model.scatter3D(GRID, density.total, weight,
                     NRand = 4000, colordim = r, axisunit = u.au,
                     cmap = 'jet', colorscale = 'log',
                     colorlabel = r'${\rm log}_{10}(r [au])$',
                     output = '3Dpoints%s.png'%tag, show = False)

#


# radmc3d image lambdarange 2600.6708541 2600.844359 nlam 20 incl 60 phi 30 setthreads 10

#from radmc3dPy.image import *
#output = 'co_cube.fits'
#dist = 140.
#im = readImage()
#im.writeFits(fname=output, dpc=dist, coord='18h10m28.652s -19d55m49.66s')
# escribiendo el cubo FITS 3D
