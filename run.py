""" Created on Sun Jun 10 15:50:48 2018
Author: Nicholas Irvin

Purpose: Execute calculations for the calculations of solar cells.

This module serves to set the parameters for the calculation. 
The module stack optimizes the stack's total power generation (which mainly handless multijunction cells),
while single_cell calculates the power of the individual cells. """

#Import libraries
import numpy as np
import input_output_management as io_management
from spectral import find_bandgaps



"""Input parameters"""
composition = ['GaAs']  # optional: Can specify material to use its absorption coefficients. Options: 'GaAs', 'Si', 'CIGS', 'CIS', 'CdTe', 'GaNP', 'perovskite triple cation'
# Alternatively, [] is generic, so the absorptance is considered to be step-function at the bandgap. 
# ['Si'] would consider all cells to be Silicon. # ['Si', 'GaNP'] would consider the second cell to be GaNP.
""" If materials aren't specified above, then just bandgaps will be specified below. If materials are specified above, then skip the next block."""
if composition == []:
        # if the material isn't specified, perform simple detailed-balance calcualtions. Then most material parameters below become irrelevant.
    sampling_range = [[1, 1]] # if materials aren't specified, then """Specify""" the bandgaps to sample for each cell.
        # Range of bandgaps to be sampled (eV).
        # Common fixed bandgaps: Si 1.125 so use sampling_range = [[1.125, 1.125]]. pc-CdTe 1.488, sc-CdTe 1.514, CIS 1.016, CIGS 1.115, https://aip-scitation-org.ezproxy1.lib.asu.edu/doi/pdf/10.1063/1.4767120?class=pdf    GaAs 1.423 Temperature dependence of semiconductor band gaps by O'Donnell
        # For a single junction, the sampling range could be [[.5,2]]  # For a double tandem, the sampling range could be [[.6, 1.1], [1.4, 2]]  # which means the first bandgap is sampled at values inbetween .6 and 1.1, and the second bandgap sampled between 1.6 and 2.     # For a triple tandem, the sampling range could be [[.3,1], [1.1,1.8], [1.8,2.6]]
        # Round the sampling_range numbers to the resolution chosen below.
    bandgap_resolution = .1  # (eV) The increment between sampled bandgap energy values.  # Choose .1 for fast calculations or .01 for normal calculations
else:  # Just let run
    sampling_range = find_bandgaps(composition)  # don't edit
    bandgap_resolution = .1


# Prepare output file.
file_name = 'Last Calculations.csv' # give calcalations file a name
io_management.create_file(file_name, sampling_range)
        

# Declare for loop so one can repeat calculations while varying parameters like doping concentrations, trap assisted lifetimes, or rear reflectance
Nd_list = np.logspace(14,20,7)  # to range doping concentrations from 10^14 to 10^20 cm^-3
for Nd in Nd_list:
    thicknesses = [3] # (list) Absorber thickness in microns. Provide as [1] or as [100] or as [1, 100] for tandems. common thicknesses are Si 300, perovskite 0.7, CdTe 3, GaAs 3
    texturing = 'No'  # 'Yes' or 'No'
   
               
    front_reflectance = 0  # between 0 and 1, probably should use 0   # External front reflectance. Reduce with antireflection coating. # can be front reflection and parasitic absorption together 
    rear_reflectance = 0  # between 0 and 1, most studies use 1 (ideal)   # proportion of photons that reflect at the rear surface
      
    dopant_type = 'n'  # 'n' or 'p'
    dopant_density = Nd # (cm^-3) default with 'undoped' at 1e10, but here we are setting the doping as the sweeping variable

    V_test = 'Voc' # Voltage at which values like radiative _lifetime may be calculated. Can be a number or the strings 'Vmp' or 'Voc'  (with the quote marks)


    trap_lifetimes = [] # (list) (units of seconds) Bulk minority trap-assisted lifetime, e.g., Shockley-Reed-Hall (SRH) lifetime. 
        # [] designates infinite (ideal) trap-assisted lifetime. [1e-9] would desinate a 1 ns trap-assisted lifetime. For multijunctions, [1e-9, 1e-6] would be designate first cell as 1 ns and second cell as 1 ms.
    SRVs = []  # (list) (cm/s) Surface recombination velocities (SRV) of rear surface, with front surface assumed ideal (0)
    
    
    
    """ Can usually ignore everything below here. """
    lifetimes = [] # (list) (s)  Minority carrier lifetime i.e. effective lifetime (s)
        # To bypass, give lifetimes as [] (default). # Intrinisc limit for undoped Silicon is 8.75 ms. # If lifetimes given as a number, then bulk_lifetimes, radiative, Auger, trap and surface recombination won't be used or calculated.
    bulk_lifetimes = [] # (list) (s) Bulk minority carrier lifetime
        # To bypass, give lifetimes as [] (default). # Good GaAs bulk_lifetime is 100 ns, great is 2,000 ns.  # If bulk lifetimes given as a number, then trap_lifetimes, radiative, Auger, trap-assisted lifetimes won't be used or calculated.            

    anything_variable = 'optional'  # Define any variable to be used in subprograms! Call on it with stack.anything_variable.
        # this was used in paper to control the mobility in carrier_models.py and also to switch from the EQE to the absorptance model in spectral.py recombination.py 
    
    diffusion_limited = 'Yes'  # 'Yes' or 'No'  # if 'Yes' then current is calculated with nonideal collection efficiency in spectral.py.      # if 'No' then standard detailed-balance procedure is used, ie  J = Jph - Jrec. 
    # diffusion_limited needs nonradiative_recombination_modeling on  
    nonradiative_recombination_modeling = 'Yes'  # 'Yes' or 'No', default 'Yes'.  # Composition should be specified. E
    if composition == []:
        nonradiative_recombination_modeling = 'No'  # Again, if material isn't speficied than many parameters besides (bandgaps and sunlight spectrum) become irrelevant
    if nonradiative_recombination_modeling == 'No':
        diffusion_limited = 'No'  # Diffusion model relies on material models
    fc_rec_ratio = 1  # >1   # ignored if nonradiative_recombination_modeling = 'Yes'. If not actually modeling nonradaitive recombination, one can directly specify the amount of nonradiative recombination. For zero nanradaitive recombination, put fc_rec_ratio = 1
    # For nonradiative recombiantion, specify either  nonradiative_recombination_modeling or fc_rec_ratio, not both
    


    spectrum = 'AM1.5G' # Solar spectrum for incident sunlight
        # Options: 'Blackbody', 'AM1.5G' (default), 'AM1.5D,' 'Laser'.      # For Laser, indictate laser energy and irradiance with ['Laser', energy, irradiance];  GaAs reportedly optimal with 1.517 laser # for example: spectrum = ['Laser', 1, 1], where energy is in eV and irradaince is in suns. 
    T_cell = 273.15 + 25  # (K) Cell temperature. (Default is 300 K or 273.15 + 25)
    T_sun = 6000  # (K) The assumed temperature of the sun. relevant only for 'Blackbody' spectrum incident light   # The Earth's irradiance data fits best to a 5778 K. # Many publications on detailed-balance calculations use 6000 K.
    concentration = 1 # Sunlight concentration. Should be between 1 (default) and 46260   # Concentration should be 1 for AM1.5G and between 2 and 46300 for AM1.5D.


    structure = 'Independent'  # Structure of the interconnnections of the multijuction cells. Options: # 'Independent' (default), 'Series', other options like 'Parallel' and 'Intermediate' need reverifying. 'Intermediate' is for intermediate-band cells.  # For that, use sampling_range = [[.6,.8], [1.1,1.4], [1.8,2.1]] or a larger range

    acceptance_angle = np.pi/2  #  (radians less than or equal to np.pi/2)   limited acceptance angles from angular selective filters or other advanced optics  # Azimuthal angle that the cell can absorb and emit into. # Minimim is 0.004649997712855405 rad  # np.pi/2 is default.   
    
    MEG = 'No'  # multi-excition generation. Choose 'Yes' or 'No' (default).
        # Need to check that absorptance extrapolates to unity with Eg = .2 eV
    # Specify the following only for cells with multi-excition generation.
    max_yield = 10**4 # ideal is over 10**3. Nonideal is less.
    threshold_energy_normalized = 2 # ideal is 2. Nonideal is more.   # 2 gives the staircase function (M_max)   # whereas 2.000001 gives a linear function (the L2)  # See Find_QY in spectrum.py for more info on MEG inputs.   
    


    """ Initiate calculations"""
    optimal_efficiency, IV_data = io_management.manage_input_output(
            file_name, T_cell, T_sun, spectrum, concentration, 
            structure, composition, V_test, fc_rec_ratio, nonradiative_recombination_modeling, diffusion_limited, lifetimes, 
            bulk_lifetimes, trap_lifetimes, SRVs, thicknesses, dopant_type, 
            dopant_density, texturing, front_reflectance, rear_reflectance, acceptance_angle, 
            MEG, max_yield, threshold_energy_normalized, sampling_range, bandgap_resolution, anything_variable)



    
