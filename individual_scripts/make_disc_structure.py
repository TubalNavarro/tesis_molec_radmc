from sf3dmodels import Model, Plot_model
import sf3dmodels.utils.units as u
import sf3dmodels.utils.constants as ct            
import sf3dmodels.rt as rt                   

from matplotlib import colors
import numpy as np
import os
import sys
import time
import json

t0 = time.time()

#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

Rout = best['intensity']['Rout']
z0 = best['height_upper']['z0']
p = best['height_upper']['p']
Rb = best['height_upper']['Rb']
q = best['height_upper']['q']

#******************
#Star pars and GRID
#******************
MStar = best['velocity']['Mstar'] * u.MSun #LkCa15 13CO
RStar = u.RSun * ( MStar/u.MSun )**0.8 
LStar = u.LSun * ( MStar/u.MSun )**4 
TStar = u.TSun * ( (LStar/u.LSun) / (RStar/u.RSun)**2 )**0.25 

sizex = sizey = sizez = 600 * u.au
nx = ny = nz = 100 
grid = Model.grid([sizex, sizey, sizez], [nx, ny, nz], rt_code='radmc3d')

#******************
#MODEL ATTRIBUTES
#******************
def z_upper_exp_tapered(coord, z0=z0, p=p, Rb=Rb, q=q, R0=100):
    R = coord['R']/u.au
    return u.au*(z0*(R/R0)**p*np.exp(-(R/Rb)**q))

density = Model.density_lyndenbell_disc(grid, H0=10*u.au, rdisc_min=30*u.au, rdisc_max=Rout*u.au)
#rdisc_min > 0 to speed up radmc photon propagation
#Rc=100*AU, Ec=30.0, gamma=1.0, H0=6.5*AU, psi=1.25, Ro=500*AU,
#rho_thres = 10.0*mH, rho_min = 2*mH,
#discFlag=True, rdisc_max = False)

density.total /= 2*ct.mH #number density
density.total[density.total == 1.0] = 0.0

temperature = Model.temperature_Powerlaw(Rout*u.au, 10, -0.5, grid, rid=1) #rid=1 takes cylindrical radius
vel = Model.velocity(None, MStar, None, density, grid, vertical=True, vel_sign=-1) #Fully Keplerian disc

#abundance = Model.abundance(5e-5, grid.NPoints)
abundance = np.zeros(grid.NPoints)
gtdratio = Model.gastodust(100., grid.NPoints)

RR = grid.rRTP[1]
zz = grid.XYZ[2]

Ru = np.unique(RR)
zw = 10.0

for Ri in Ru:
    zR = z_upper_exp_tapered({'R': Ri})
    ind_pos = ((RR==Ri) & (zz >= zR-zw*u.au) & (zz <= zR+zw*u.au))
    ind_neg = ((RR==Ri) & (zz >= -zR-zw*u.au) & (zz <= -zR+zw*u.au))
    ind = ind_pos | ind_neg
    abundance[ind] = 5e-7
    
#**********************
#WRITE RADMC-3D FILES
#**********************
prop = {'dens_H2': density.total,
        'dens_dust': 2*ct.mH * density.total * 1/gtdratio, #mass density
        'temp_dust': temperature.total, #30+np.zeros_like(density.total),
        'velocity': [vel.x, vel.y, vel.z],
        'gtdratio': gtdratio,
        'microturbulence': 100.+np.zeros(grid.NPoints),
        'abundance': abundance} #*1e-3 to make it opt. thin

radmc = rt.Radmc3d(grid)
wavelength_intervals = [1e-1,5e2,1e4] #[5e-3, 5e1, 1e4]
wavelength_divisions = [20,20] 
radmc.write_radmc3d_control(nphot=100000000, incl_dust=1, setthreads=8, incl_freefree=0, tgas_eq_tdust=1, modified_random_walk=1)
radmc.write_amr_grid()
radmc.write_dust_density(prop['dens_dust']) #Mass density
radmc.write_dust_temperature(prop['temp_dust']) #Spherical radial plaw temperature produces artifacts near the disc outer radius in the line images
radmc.write_gas_velocity(prop['velocity'])
radmc.write_microturbulence(prop['microturbulence'])
radmc.write_stars(nstars=1, pos=[[0,0,0]], rstars = [u.RSun], mstars = [MStar], flux = [[-u.TSun]], #flux --> if negative, radmc assumes the input number as the blackbody temperature of the star 
                  lam = wavelength_intervals, nxx = wavelength_divisions) 
radmc.write_wavelength_micron(lam = wavelength_intervals, nxx = wavelength_divisions) #lam --> wavelengths in microns, nxx --> number of divisions in between wavelengths

#********************************
#Write CO number density file
#********************************
nx,ny,nz = grid.Nodes

with open('numberdens_co.inp','w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n'%(nx*ny*nz))           # Nr of cells
    data = prop['abundance']*density.total*(100**-3) #To 1/cm3 units
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')

#Lines file
with open('lines.inp','w+') as f:
    f.write('2\n')
    f.write('1\n')
    f.write('co    leiden    0    0    0\n')

print ('Ellapsed time: %.3fs' % (time.time() - t0))
print ('-------------------------------------------------\n-------------------------------------------------\n')
    
#*****************************************
#DIAGNOSTIC PLOTS, TOTAL MASS AND MEAN T
#*****************************************
Model.PrintProperties(density, temperature, grid)

Plot_model.plane2D(grid, prop['dens_H2'], axisunit=u.au, plane = {'y': 0}, norm=colors.LogNorm(vmin=1e9, vmax=1e13), output='densH2_2D.png')
Plot_model.plane2D(grid, prop['abundance'], axisunit=u.au, plane = {'y': 0}, norm=colors.LogNorm(vmin=1e-9, vmax=1e-6), output='abund_2D.png')
Plot_model.plane2D(grid, prop['temp_dust'], axisunit=u.au, plane = {'y': 0}, norm=colors.LogNorm(vmin=1e1, vmax=1e3), output='tempgas_2D.png')

