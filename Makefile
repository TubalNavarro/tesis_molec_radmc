cleanmodel:
	@rm -f	*.o *.mod *.dat *.out *.info *.used                 \
	   *wavelength_micron.inp dustopac.inp dust_density.inp     \
	   dust_density.uinp *.uout *.udat                          \
	   amr_grid.inp stars.inp radmc3d.inp
	@echo All .o, .mod, .out files and most .inp files removed.

cleanall:
	@rm -f	*.o *.mod *.pyc *.dat *.out *.info *.used *.png     \
	   *wavelength_micron.inp dustopac.inp dust_density.inp     \
	   dust_density.uinp *.uout *.udat                          \
	   amr_grid.inp stars.inp radmc3d.inp image_script.pro      \
	   radmc3d Makefile~ *.pro~ README*~ *.f90~ *.inp~ *.py~    \
	   microturbulence.inp number_density.inp gas_temperature.inp \
	   numberdens_co.inp lines.inp gas_velocity.inp cube.fits
	@rm -r model_*
	@echo Directory cleaned to basic
cleanline:
	@rm -f	*.out *.fits
	@echo .out and .fits files removed.
