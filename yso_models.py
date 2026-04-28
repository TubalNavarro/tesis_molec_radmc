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
from plot_helpers import *
#-----------------
#Extra libraries
#-----------------
from matplotlib import colors
import numpy as np
import os
import time


    #********************************
    #Write molecule number density file
    #********************************

def write_molecule_files(nx,ny,nz, density, prop, molec=''):
    with open('numberdens_%s.inp'%molec,'w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nx*ny*nz))           # Nr of cells
        data = prop['abundance']*density.total*(100**-3) #To 1/cm3 units
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')

    #Lines file
    with open('lines.inp','w+') as f:
        f.write('2\n')
        f.write('1\n')
        f.write('%s    leiden    0    0    0\n'%molec)

    #
    # Dust opacity control file
    #
    with open('dustopac.inp','w+') as f:
        f.write('2               Format number of this file\n')
        f.write('1               Nr of dust species\n')
        f.write('============================================================================\n')
        f.write('1               Way in which this dust species is read\n')
        f.write('0               0=Thermal grain\n')
        f.write('silicate        Extension of name of dustkappa_***.inp file\n')
        f.write('----------------------------------------------------------------------------\n')
    

def UlrichDisk(discFlag=True, cavity_ang=60 ,envFlag=True, nmodel=0, MStar=20, MRate=1e-3, Rdisc=700, Arho0=10, Renv=8000, exp_disc=2.25, prop_only=False, molec='co', molec_abund=7.5e-7, const_T=300):

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
    #LStar = u.LSun * ( MStar/u.MSun )**4  #L propto M**4?
    
    #-------------------------------
    #Parameters for the Pringle disc
    #-------------------------------
    MRate = MRate * u.MSun_yr
    RStar = 10*u.RSun * ( MStar/u.MSun )**0.8  #????

    #RStar = 26 * u.RSun * ( MStar/u.MSun )**0.27 * ( MRate / (1e-3*u.MSun_yr) )**0.41

    LStar=  1e4*u.Lsun

    print('RStar:'.format(RStar))
    TStar = u.TSun * ( (LStar/u.LSun) / (RStar/u.RSun)**2 )**0.25
    Rd = Rdisc * u.au

    print ('RStar:', RStar/u.RSun,', LStar:', LStar/u.LSun, ', TStar:', TStar)
    
    #---------------
    #GRID Definition
    #---------------
    #Cubic grid, each edge ranges [-size, size] au.

    sizex = sizey = sizez = 1584 * u.au #half size
    Nx = Ny = Nz = 99 #Number of divisions for each axis
    GRID = Model.grid([sizex, sizey, sizez], [Nx, Ny, Nz], rt_code = 'radmc3d', include_zero = True)
    NPoints = GRID.NPoints #Final number of nodes in the grid
  

    #--------
    #DENSITY
    #--------
    Rho0 = Res.Rho0(MRate, Rd, MStar)
    Arho = Arho0 #Disc-envelope density factor
    Renv = Renv * u.au #Envelope radius
    Cavity = cavity_ang * np.pi/180 #Cavity opening angle 
    density = Model.density_Env_Disc(RStar, Rd, Rho0, Arho, GRID, exp_disc=exp_disc, 
                                     discFlag = discFlag, envFlag = envFlag, rho_min_env=1.0e7,
                                     renv_max = Renv, ang_cavity = Cavity, 
                                     average_around_Rd=np.median)


    #---------------------
    # MODEL TEMPERATURE
    #---------------------
    T10Env=1200
    BT = 10
    temperature = Model.temperature(TStar, Rd,T10Env, RStar, MStar, MRate, BT, density, GRID)
    #temperature=Model.temperature_Constant(density, GRID, discTemp = 2.5*const_T, envTemp = const_T, backTemp = 30.0)
    
    #Whitney et al. exponent is p=0.33 (in Keto & Zhang (2/(4+p)) where p<-1  )
   
    #--------
    #VELOCITY
    #--------
    vel = Model.velocity(RStar, MStar, Rd, density, GRID)
    
#    #-----------------------------------------------
#    #3D Points Distribution (weighting with density)
#    #-----------------------------------------------
#    tag = 'Main'
#    dens_plot = density.total / 1e6
#
#    weight = 10*Rho0
#    r = GRID.rRTP[0] / u.au #GRID.rRTP hosts [r, R, Theta, Phi] --> Polar GRID
#    Plot_model.scatter3D(GRID, density.total, weight,
#                     NRand = 4000, colordim = r, axisunit = u.au,
#                     cmap = 'jet', colorscale = 'log',
#                     colorlabel = r'${\rm log}_{10}(r [au])$',
#                     output = '3Dpoints%s.png'%tag, show = False)
#
#---------------------
#2D PLOTTING (Density)
##---------------------
#
#    vmin, vmax = np.array([2e10, 5e19]) / 1e6
#    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
#
#    Plot_model.plane2D(GRID, dens_plot, axisunit = u.au,
#                       cmap = 'jet', plane = {'z': 0*u.au},
#                       norm = norm, colorlabel = r'$[\rm cm^{-3}]$',
#                       output = 'DensMidplane_%s.png'%tag, show = False)
#
#    vmin, vmax = np.array([2e10, 5e19]) / 1e6
#    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
#
#    Plot_model.plane2D(GRID, dens_plot, axisunit = u.au,
#                       cmap = 'jet', plane = {'y': 0*u.au},
#                       norm = norm, colorlabel = r'$[\rm cm^{-3}]$',
#                       output = 'DensVertical_%s.png'%tag, show = False)
#
#    #---------------------
#    #2D PLOTTING (Temp)
#    #---------------------
#
#    vmin, vmax = np.array([5e1, 1e4])
#    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
#
#    Plot_model.plane2D(GRID, temperature.total, axisunit = u.au,
#                       cmap = 'jet', plane = {'z': 0*u.au},
#                       norm = norm, colorlabel = r'[Kelvin]',
#                       output = 'TempMidplane_%s.png'%tag, show = False)
#
#
#    vmin, vmax = np.array([5e1, 1e4])
#    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
#
#    Plot_model.plane2D(GRID, temperature.total, axisunit = u.au,
#                       cmap = 'jet', plane = {'y': 0*u.au},
#                       norm = norm, colorlabel = r'[Kelvin]',
#                       output = 'TempVertical_%s.png'%tag, show = False)
#    
#    #---------------------
#    #2D PLOTTING (Emissivity)
#    #---------------------
#    vmin, vmax = np.array([3e7, 5e12])
#    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
#
#    Plot_model.plane2D(GRID, temperature.total * dens_plot, axisunit = u.au,
#                       cmap = 'ocean_r', plane = {'y': 0*u.au},
#                       norm = norm, colorlabel = r'[$\rho$ T]',
#                       output = 'Emissivity_%s.png'%tag, show = False)
#
    #**********************
    #WRITE RADMC-3D FILES
    #**********************
    abundance = molec_abund+np.zeros(GRID.NPoints) #Optimize for the molecule
    gtdratio = Model.gastodust(100., GRID.NPoints)
    microturb = 100+np.zeros(GRID.NPoints)
    prop = {'dens_H2': density.total,
        'dens_dust': 2*ct.mH * density.total * 1/gtdratio, #mass density # ct.mH -> u.amu
        'temp_dust': temperature.total, #30+np.zeros_like(density.total),
        #'temp_gas': temperature.total, 
        'velocity': [vel.x, vel.y, vel.z],
        'gtdratio': gtdratio,
        'microturbulence': microturb,
        'abundance': abundance}

    if prop_only: return GRID, prop, density
    
    radmc = rt.Radmc3d(GRID)
    wavelength_intervals = [1e-1,5e2,1e4] #[5e-3, 5e1, 1e4]
    wavelength_divisions = [20,20] 
    radmc.write_radmc3d_control(nphot=100000000, incl_dust=1, setthreads=8, incl_freefree=0, tgas_eq_tdust=1, modified_random_walk=1, catch_doppler_resolution=0.2)
    radmc.write_amr_grid()
    radmc.write_dust_density(prop['dens_dust']) #Mass density
    radmc.write_dust_temperature(prop['temp_dust']) #Spherical radial plaw temperature produces artifacts near the disc outer radius in the line images
    radmc.write_gas_velocity(prop['velocity'])
    radmc.write_microturbulence(prop['microturbulence'])
    radmc.write_stars(nstars=1, pos=[[0,0,0]], rstars = [u.RSun], mstars = [MStar], flux = [[-u.TSun]], #flux --> if negative, radmc assumes the input number as the blackbody temperature of the star 
                  lam = wavelength_intervals, nxx = wavelength_divisions) 
    radmc.write_wavelength_micron(lam = wavelength_intervals, nxx = wavelength_divisions) #lam --> wavelengths in microns, nxx --> number of divisions in between wavelengths
    nx,ny,nz = GRID.Nodes

    write_molecule_files(nx=nx,ny=ny,nz=nz, density=density, prop=prop, molec=molec)

    #plot_1d_props(GRID, prop, density, tag='')
    #-------
    #TIMING
    #-------
    print ('Ellapsed time for iterarion of create_model.py: %.3fs' % (time.time() - t0))
    print ('-------------------------------------------------\n-------------------------------------------------\n')

    

def Hamburguers(nmodel=0, MStar=20, MRate=1e-4, discFlag = True, Rdisc=750, Arho0=10, 
prop_only=False, molec='co', molec_abund=5e-7):

#-------
#DISC
#-------
    H0sf = 0.03 #Disc scale height factor (H0 = H0sf * RStar)
    Arho = 5.25 #Disc density factor
   
    t0 = time.time()

    print('\n')
    print('Passed paremeters are:')
    print('nmodel: {}'.format(nmodel))
    print('MStar: {}'.format(MStar))
    print('MRate: {}'.format(MRate))
    print('Rdisc: {}'.format(Rdisc))
    print('Arho0: {}'.format(Arho0))
    #print('Renv: {}'.format(Renv))
    #------------------
    MStar = MStar * u.MSun
    LStar = u.LSun * ( MStar/u.MSun )**4
    
    #-------------------------------
    #Parameters for the Pringle disc
    #-------------------------------
    MRate = MRate * u.MSun_yr
    RStar = u.RSun * ( MStar/u.MSun )**0.8
    print('RStar:'.format(RStar))
    TStar = u.TSun * ( (LStar/u.LSun) / (RStar/u.RSun)**2 )**0.25
    Rd = Rdisc * u.au

    print ('RStar:', RStar/u.RSun,', LStar:', LStar/u.LSun, ', TStar:', TStar)
    
    #---------------
    #GRID Definition
    #---------------
    #Cubic grid, each edge ranges [-size, size] au.

    sizex = sizey = sizez = 4000 * u.au
    Nx = Ny = Nz = 80 #Number of divisions for each axis
    GRID = Model.grid([sizex, sizey, sizez], [Nx, Ny, Nz], rt_code = 'radmc3d', include_zero = True)
    NPoints = GRID.NPoints #Final number of nodes in the grid
  

    #--------
    #DENSITY
    #--------
    Rho0 = Res.Rho0(MRate, Rd, MStar)
    print('#####%f######'%(Rho0))
    density = Model.density_Hamburgers(RStar, H0sf, Rd, Rho0, Arho, GRID,
                                    discFlag = True, rdisc_max = Rd*1.5)


    #---------------------
    # MODEL TEMPERATURE
    #---------------------
    # Ionized gas temperature

    t_e = 1.0e2 #K


    temperature = Model.temperature_Constant(density, GRID, discTemp=t_e, backTemp=2.725480)

    
   
    #--------
    #VELOCITY
    #--------
    vel = Model.velocity(RStar, MStar, Rd, density, GRID)
#(IOnized gas)
    Model.PrintProperties(density, temperature, GRID, species='dens_ion')
    Model.PrintProperties(density, temperature, GRID, species='dens_e')
    
    
    #-------------------------
    #ROTATION, VSYS, CENTERING
    #-------------------------

    #xc, yc, zc = [0.0,0.0,0.0]
    #CENTER = [xc, yc, zc] #New center of the region in the global grid
    #newProperties = Model.ChangeGeometry(GRID, center = CENTER,  vel = vel,
    	      	 	             #rot_dict = {'angles': [0*(np.pi/2)*(60./90)], 'axis': ['x'] })
    #GRID.XYZ = newProperties.newXYZ #XYZ redefinition
    #vel.x, vel.y, vel.z = newProperties.newVEL #vels redefinition

    #rot_dict = {'angles': [(np.pi/2)*(135./90),(np.pi/2)*(60./90)], 'axis': ['y','x'] })


    #**********************
    #WRITE RADMC-3D FILES
    #**********************
    abundance = molec_abund+np.zeros(GRID.NPoints) #Optimize for the molecule
    gtdratio = Model.gastodust(100., GRID.NPoints)
    microturb = 100.+np.zeros(GRID.NPoints)
    prop = {'dens_H2': density.total,
        'dens_dust': 2*ct.mH * density.total * 1/gtdratio, #mass density # ct.mH -> u.amu
        'temp_dust': temperature.total, #30+np.zeros_like(density.total),
        #'temp_gas': temperature.total, 
        'velocity': [vel.x, vel.y, vel.z],
        'gtdratio': gtdratio,
        'microturbulence': microturb,
        'abundance': abundance}

    if prop_only: return GRID, prop, density
    
    radmc = rt.Radmc3d(GRID)
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
    nx,ny,nz = GRID.Nodes
    
    write_molecule_files(nx=nx,ny=ny,nz=nz, density=density, prop=prop, molec=molec)

    
   
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

    #-------
    #TIMING
    #-------
    print ('Ellapsed time for iterarion of create_model.py: %.3fs' % (time.time() - t0))
    print ('-------------------------------------------------\n-------------------------------------------------\n')

    

def Hamburguers_piecewise(nmodel=0, MStar=20, MRate=1e-4, discFlag = True, Rdisc=150, Arho0=10, prop_only=False, molec='co' ,molec_abund=5e-7):


#def density_Hamburgers_piecewise(RStar, H0, R_list, p_list, rho0, GRID, RH_list = None,
#                               q_list = [0.5], rho_thres = 10.0, rho_min = 0.0, 
 #                                Rt = False):

#RStar: stellar radius
#H0: scaleheight normalization constant --> usually H0 = shFactor * R0
#R_list: List of polar limits, length (n,)
#p_list: List of powerlaws in R_list intervals, length (n-1,)
#rho0: density at R_list[0]
#Optionals:
#RH_list: List of limits for piecewise scaleheight, length (n,)
#q_list: List of powerlaws in RH_list intervals, length (n-1,)
#rho_thres: minimum reachable density by the model
#rho_min: background density
#Rt: radius where the disc tapering starts



#-------
#DISC
#-------
    H0sf = 0.03 #Disc scale height factor (H0 = H0sf * RStar)
    Arho = 5.25 #Disc density factor
   
    t0 = time.time()

    print('\n')
    print('Passed paremeters are:')
    print('nmodel: {}'.format(nmodel))
    print('MStar: {}'.format(MStar))
    print('MRate: {}'.format(MRate))
    print('Rdisc: {}'.format(Rdisc))
    print('Arho0: {}'.format(Arho0))
    #print('Renv: {}'.format(Renv))
    #------------------
    MStar = MStar * u.MSun
    LStar = u.LSun * ( MStar/u.MSun )**4
    
    #-------------------------------
    #Parameters for the Pringle disc
    #-------------------------------
    MRate = MRate * u.MSun_yr
    RStar = u.RSun * ( MStar/u.MSun )**0.8
    print('RStar:'.format(RStar))
    TStar = u.TSun * ( (LStar/u.LSun) / (RStar/u.RSun)**2 )**0.25
    Rd = Rdisc * u.au

    print ('RStar:', RStar/u.RSun,', LStar:', LStar/u.LSun, ', TStar:', TStar)
    
    #---------------
    #GRID Definition
    #---------------
    #Cubic grid, each edge ranges [-size, size] au.

    sizex = sizey = sizez = 4000 * u.au
    Nx = Ny = Nz = 80 #Number of divisions for each axis
    GRID = Model.grid([sizex, sizey, sizez], [Nx, Ny, Nz], rt_code = 'radmc3d', include_zero = True)
    NPoints = GRID.NPoints #Final number of nodes in the grid
  

    #--------
    #DENSITY
    #--------
    Rho0 = Res.Rho0(MRate, Rd, MStar)
    print('#####%f######'%(Rho0))
    density = Model.density_Hamburgers_piecewise(RStar, RStar*H0sf, [Rd/1000,Rd/3, 2*Rd/3,Rd],[1,0.5,0.2], Rho0, GRID)

    #---------------------
    # MODEL TEMPERATURE
    #---------------------
    # Ionized gas temperature

    t_e = 100 #K


    temperature = Model.temperature_Constant(density, GRID, discTemp=t_e, backTemp=2.725480)

    
   
    #--------
    #VELOCITY
    #--------
    vel = Model.velocity(RStar, MStar, Rd, density, GRID)
#(IOnized gas)
    Model.PrintProperties(density, temperature, GRID, species='dens_ion')
    Model.PrintProperties(density, temperature, GRID, species='dens_e')
    
    
    #-------------------------
    #ROTATION, VSYS, CENTERING
    #-------------------------

    #xc, yc, zc = [0.0,0.0,0.0]
    #CENTER = [xc, yc, zc] #New center of the region in the global grid
    #newProperties = Model.ChangeGeometry(GRID, center = CENTER,  vel = vel,
    	      	 	             #rot_dict = {'angles': [0*(np.pi/2)*(60./90)], 'axis': ['x'] })
    #GRID.XYZ = newProperties.newXYZ #XYZ redefinition
    #vel.x, vel.y, vel.z = newProperties.newVEL #vels redefinition

    #rot_dict = {'angles': [(np.pi/2)*(135./90),(np.pi/2)*(60./90)], 'axis': ['y','x'] })


    #**********************
    #WRITE RADMC-3D FILES
    #**********************
    abundance = molec_abund+np.zeros(GRID.NPoints) #Optimize for the molecule
    gtdratio = Model.gastodust(100., GRID.NPoints)
    microturb = 100.+np.zeros(GRID.NPoints)
    prop = {'dens_H2': density.total,
        'dens_dust': 2*ct.mH * density.total * 1/gtdratio, #mass density # ct.mH -> u.amu
        'temp_dust': temperature.total, #30+np.zeros_like(density.total),
        #'temp_gas': temperature.total, 
        'velocity': [vel.x, vel.y, vel.z],
        'gtdratio': gtdratio,
        'microturbulence': microturb,
        'abundance': abundance}

    if prop_only: return GRID, prop, density
    
    radmc = rt.Radmc3d(GRID)
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
    nx,ny,nz = GRID.Nodes
    
    write_molecule_files(nx=nx,ny=ny,nz=nz, density=density, prop=prop, molec=molec)

    
   
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

    #-------
    #TIMING
    #-------
    print ('Ellapsed time for iterarion of create_model.py: %.3fs' % (time.time() - t0))
    print ('-------------------------------------------------\n-------------------------------------------------\n')



