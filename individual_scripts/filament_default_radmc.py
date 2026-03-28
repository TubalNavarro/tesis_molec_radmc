import sf3dmodels.filament as sf
import sf3dmodels.Plot_model as pm
from sf3dmodels.utils.units import pc
import sf3dmodels.rt as rt

import sf3dmodels.utils.constants as ct   
import numpy as np         


f1 = sf.FilamentModel([0,0,0], [0,0,1], -0.3*pc, 0.3*pc, 0.01*pc)
f1.cylinder(0.15*pc, 1e-3*pc, temp_pars = [50, 0.02*pc, -0.3], 
            vel_pars = [(0,0.02*pc,0),-2500.,0],
            abund_pars = 1e-10, dummy_frac=0.3)

lims=[-0.4*pc,0.4*pc]
#pm.scatter3D(f1.GRID, f1.density, f1.density.min(), axisunit = pc,
#             colordim = f1.temperature, 
#             colorlabel = 'T [K]',
#             xlim=lims, ylim=lims, zlim=lims,
#             NRand = 5000, show=True)

#prop = {'dens_H': f1.density,
#        'temp_gas': f1.temperature,
#        'abundance': f1.abundance,
#        'gtdratio': f1.gtdratio,
#        'vel_x': f1.vel.x,
#        'vel_y': f1.vel.y,
#        'vel_z': f1.vel.z,
#        }

#**********************
#WRITE RADMC-3D FILES
#**********************
gtdratio = np.full(f1.grid.size,100)
microturb = 100.+np.zeros(f1.grid.size)

prop = {'dens_H2': f1.density,
        'dens_dust': 2*ct.mH * f1.density * 1/gtdratio, #mass density
        'temp_dust': f1.temperature, #30+np.zeros_like(density.total),
        'velocity': [f1.vel.x, f1.vel.y, f1.vel.z],
        'gtdratio': gtdratio,
        'microturbulence': microturb,
        'abundance': f1.abundance} #*1e-3 to make it opt. thin

#lime = rt.Lime(f1.GRID)
#lime.submodel(prop, output='datatab.dat', folder='./', lime_header=True, lime_npoints=True)



radmc = rt.Radmc3d(f1.GRID)
