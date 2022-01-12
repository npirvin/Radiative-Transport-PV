"""
This module defines some useful objects and also does some boring clerical tasks for the detailed-balance calculations.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import csv
import stack_power, spectral


def create_file(file_name, sampling_range):
    file = open(file_name, 'w+', newline='') # Check that file is openable and empty
    writer = csv.writer(file)
    if len(sampling_range) != 1:
        writer.writerow(['Bandgap (eV)' for i in range(len(sampling_range))] + ['Efficiency (%)']                     
            + ['Cell Temperature (K)'] + ['Sun Temperature (K)'] + ['Sunlight Spectrum']  + ['Sunlight Concentration (Suns)'] 
            + ['Tandem Connections'] + ['Acceptance Angle (rad)'] + ['Texturing']  + ['Backsheet Reflectance'] + ['Composition']
            + ['Nonradiative Recombination'] + ['Layer Thickness (um)'] + ['Dopant Density (cm^-3)'] + ['Dopant Type'])
    else:
        writer.writerow(['Bandgap (eV)'] + ['Efficiency (%)'] 
            + ['Jsc (mA/cm$^2$)'] + ['Voc (V)'] + ['FF'] + ['Jmp (mA/cm^2)'] + ['Vmp (V)'] + ['Jph at Vmp (mA/cm^2)']           
            + ['Test Voltage (V) for following thru Vdb'] + ['Jph at V_test (mA/cm^2)']  + ['dn (cm^-3, at test Voltage)'] 
            + ['J_Auger (mA/cm^2)'] + ['J_rad_front (mA/cm^2)'] + ['J_rad_back (mA/cm^2)'] + ['J_FCA (mA/cm^2)']
            + ['J_trap (mA/cm^2)'] + ['J_SRV (mA/cm^2)'] + ['External Radiative Efficiency'] + ['Photon Recycling Probability']   
            + ['Vdb, Detailed-Balance Voc (V)'] + ['Cell Temperature (K)'] + ['Sun Temperature (K)'] 
            + ['Sunlight Spectrum']  + ['Sunlight Concentration (Suns)'] 
            + ['Structure'] + ['Acceptance Angle (rad)'] + ['Texturing']  + ['Rear Reflectance'] + ['Composition']
            + ['Nonradiative Recombination'] + ['Lifetime Effective (s)'] + ['trap_lifetime (s)'] 
            + ['Radiative Lifetime (s)'] + ['Auger Lifetime (s)'] + ['SRVs'] 
            + ['Effective Diffusion Length (um)'] + ['Electrical Diffusion Length (um)'] + ['Ideal Electrical Diffusion Length (um)']
            + ['Dumke Dpr (cm^2/s)'] + ['Dumke Lpr (um)'] + ['Hypothetical Jrad_abs (mA/cm^2)']
            + ['Layer Thickness (um)'] + ['Dopant Density (cm^-3)'] 
            + ['Dopant Type'] + ['Anything Variable (made to change from run to run)']  + ['Anything Output'])
    return True


class Stack:
    """ Major program data object! Stores data into a littl packet.
    This object stack combines attributes of the solar cell stack and its environment.
    Later we add stored data like the current layer of interest, then you can access that stack.layer_num.
    Stack initiated by function manage_results"""

    def __init__(self, T_cell, T_sun, spectrum, concentration, structure, composition,
                  V_test, fc_rec_ratio, nonradiative_recombination_modeling, diffusion_limited, lifetimes, bulk_lifetimes, trap_lifetimes, 
                  SRVs, thicknesses, dopant_type, dopant_density, 
                  texturing, front_reflectance, rear_reflectance, 
                  acceptance_angle, MEG, max_yield, energy_threshold_normalized, 
                  sampling_range, anything_variable):
        self.T_cell = T_cell
        self.T_sun = T_sun
        self.spectrum = spectrum
        self.concentration = concentration
        self.structure = structure
        self.composition = composition
        self.V_test = V_test
        self.fc_rec_ratio = fc_rec_ratio
        self.nonradiative_recombination_modeling = nonradiative_recombination_modeling
        self.diffusion_limited = diffusion_limited
        self.lifetimes = lifetimes
        self.bulk_lifetimes = bulk_lifetimes
        self.trap_lifetimes = trap_lifetimes
        self.SRVs = SRVs
        self.dopant_type = dopant_type
        self.dopant_density = dopant_density
        self.texturing = texturing
        self.front_reflectance = front_reflectance
        self.rear_reflectance = rear_reflectance
        self.acceptance_angle =  acceptance_angle
        self.MEG = MEG      
        self.max_yield = max_yield
        self.energy_threshold_normalized = energy_threshold_normalized
       
        self.number_of_bandgaps = len(sampling_range)
        self.layer_num = 0
        self.SRVs = SRVs
        self.anything_variable = anything_variable
        self.counter = 0  # for counting number of times a def is called
        self.voltage_dependent_Jgen = 'No'  # Turn on to recalculate Jsc at different voltages
        thicknesses = np.array(thicknesses)*1e-4  # Convert from microns to cm
        self.thicknesses = thicknesses
        self.number_of_bandgaps = len(sampling_range)
        self.layer_num = 0
        self.counter = 0  # for counting number of times a def is called
        self.voltage_dependent_Jgen = 'On'
        self.rec_data = 'N/A'  
        self.extra_plots = 'No'
        self.anything_output = 0
        self.P_PR = 'N/A'

    def set_layer(self, stack, value):
       # Value is 0 for the bottom absorber layer and is number_of_bandgaps-1 for the top layer
       self.layer_num = value
       self.thickness = stack.thicknesses[value] if(np.shape(stack.thicknesses) != (0,)) else 'N/A'
       inf = float('inf')
       self.lifetime = stack.lifetimes[value] if(np.shape(stack.lifetimes) != (0,)) else inf
       self.bulk_lifetime = stack.bulk_lifetimes[value] if(np.shape(stack.bulk_lifetimes) != (0,)) else inf
       self.trap_lifetime = stack.trap_lifetimes[value] if(np.shape(stack.trap_lifetimes) != (0,)) else inf
       self.SRV = stack.SRVs[value] if(np.shape(stack.SRVs) != (0,)) else 0
    def set_bandgaps(self, bandgaps):
       # Value is 0 for the bottom absorber layer and is number_of_bandgaps-1 for the top layer
       self.bandgaps = bandgaps      
    def set_bandgap(self, bandgap):
       # Value is 0 for the bottom absorber layer and is number_of_bandgaps-1 for the top layer
       self.bandgap = bandgap
    def increment_counter(self):
       self.counter = self.counter + 1
    def change_variable(self, variable_name, new_value):
       self.variable_name = new_value
       
    def put_voltage_dependent_Jgen(self, value):
       self.voltage_dependent_Jgen = value
       
    def switch_MEG_to(self, value):
      # Value is 'Yes' or 'No'
      self.MEG = value               



class SampledBandgaps:
    """This object sampled_bandgaps combines information together on which bandgap
    energies will be sampled.
    Used by manage_results"""
    
    def __init__(self, sampling_range, bandgap_resolution, spectrum):
        # Set a large number, 'infinity', to be used as the upper energy limit
        infinity = 10 if(spectrum=='Blackbody') else 4.428
        # 10 eV is large for temperatures under 6000, 
        # but the AM1.5 data only goes to 4.428
        self.infinity = infinity

        self.spectrum = spectrum
        self.sampling_range = sampling_range
        self.lower_limits = [cell[0] for cell in sampling_range] # + [infinity]
        # Bandgap range minimum for each cell
        self.upper_limits = [cell[1] for cell in sampling_range] # + [infinity]
        # Bandgap range maximum for each cell
        self.bandgap_resolution = bandgap_resolution
                
        # Create the array that stores what bandgaps will be tested.
        energies = np.round(np.arange(sampling_range[0][0], sampling_range[-1][-1]+2*bandgap_resolution, bandgap_resolution), 5)
        # "+2*bandgap_resolution" because arange doesn't include the endpoint, 
        # and we also want room for infinity.
        energies[-2] = sampling_range[-1][-1]  # Make sure to end with upper bound. 
        energies[-1] = infinity  # Set the last energy to cover the whole energy range.
        self.energies = energies

 
# if varying bandgap then make plots about that
def make_plots(optimal_bandgaps, file_data, sampled_bandgaps, stack):
    """ Plot bottom bandgap vs efficiency.
    One can also use the plot_from_file function in auxillaries.py to plot without
    recalculating results.
    Used by manage_input_output."""
    
    if plt.fignum_exists(2):  # Already made a plot
        pass
    else:
        if sampled_bandgaps.lower_limits[0] == sampled_bandgaps.upper_limits[0]:
            return True
        plt.figure()
        
        if stack.number_of_bandgaps == 1:
            # Plot one easy curve.
            plt.plot([line[0] for line in file_data], [line[1] for line in file_data])
            plt.xlabel('Bottom-Cell Bandgap (eV)')
            plt.ylabel('Maximum Efficiency (%)')
        
        elif (stack.number_of_bandgaps == 2 and 
              sampled_bandgaps.lower_limits[0] == sampled_bandgaps.upper_limits[0]):
            # Fixed bottom bandgap so plot one easy curve.
            plt.plot([line[1] for line in file_data], [line[2] for line in file_data])
            plt.xlabel('Top-Cell Bandgap (eV)')
            plt.ylabel('Maximum Efficiency (%)')
        elif stack.structure == "Intermediate":
            plt.scatter([line[2] for line in file_data], [line[3] for line in file_data])
            plt.xlabel('Full Bandgap (eV)')
            plt.ylabel('Maximum Efficiency (%)')    
     
        else:  # Plot multiple curves for various top bandgaps

            # List of bottom bandgaps sampled:
            bottom_bandgaps = np.unique([line[0] for line in file_data])
        
            def fixed_top_bandgap_data(top_bandgap):
                """Filter the file_data for configurations with the top bandgap equal
                to the specified value. Also only take the best efficiency for each 
                bottom bandgap value, which selects the optimal middle bandgaps."""

                # Filter data with top_bandgap of specified value
                data = [line[0:stack.number_of_bandgaps+1] for index, line in enumerate(file_data) 
                        if line[stack.number_of_bandgaps-1] == top_bandgap]
                
                # Group data by bottom bandgap...
                data = [[line for index, line in enumerate(data) if line[0] == bottom_bandgap] 
                        for bottom_bandgap in bottom_bandgaps 
                        if bottom_bandgap <= top_bandgap]
                # and select the optimal configuration in each group.
                data = [max(group, key=lambda x: x[-1]) for group in data]
                
                # Output the bottom bandgaps and efficiencies in a savable format.
                x = [line[0] for line in data]  # Bottom bandgaps
                y = [line[-1] for line in data]  # Efficiencies
                return(x, y)
                
            # Prepare plot data for the optimal top bandgap as well as the top bandgap's
            # lower and upper limits.
            top_bandgap1 = optimal_bandgaps[-1] 
            top_bandgap2 = sampled_bandgaps.lower_limits[-1]
            top_bandgap3 = sampled_bandgaps.upper_limits[-1]
            x1, y1 = fixed_top_bandgap_data(top_bandgap1)
            x2, y2 = fixed_top_bandgap_data(top_bandgap2)
            x3, y3 = fixed_top_bandgap_data(top_bandgap3)
    
            # Create plots
            plt.rc('font', family='sans-serif')
            fig1, ax1 = plt.subplots()
            if top_bandgap1 != top_bandgap2 and top_bandgap1 != top_bandgap3:
                # Check for redundancies
                plt.plot(x1, y1, label='Top Bandgap = ' + str(top_bandgap1) + ' eV')
            plt.plot(x2, y2, label='Top Bandgap = ' + str(top_bandgap2) + ' eV')
            plt.plot(x3, y3, label='Top Bandgap = ' + str(top_bandgap3) + ' eV')
            plt.xlabel('Bottom Cell\'s Bandgap (eV)')
            plt.ylabel('Maximum Efficiency (%)')
            plt.legend()
        
        if stack.number_of_bandgaps == 3 and stack.structure != "Intermediate":
            print("In the plot below, each point has a specified bottom and top bandgaps."
                  "The middle cell's bandgap is the one that gives the optimal efficiency.")
        elif stack.number_of_bandgaps > 3:
            print("In the plot below, each point has a specified bottom and top bandgaps."
                  " The middle cells\'s bandgaps are those that give the optimal efficiency.")     
    return True
    
    


def write_file(file_data, file_name, stack):
    """Write the file that stores the efficiency at every sampled bandgap configuration.
    Used by manage_input_output."""
    
    file = open(file_name, 'a+', newline='')
    writer = csv.writer(file)
    for line in file_data:
        writer.writerow(line)   
    file.close()
    return True



def display_results(start_time, optimal_efficiency, optimal_bandgaps, file_name, stack):   
    """ Display main results for manage_input_output."""
    if stack.number_of_bandgaps == 1:
        print('The maximum efficiency is ' + str(round(optimal_efficiency, 3)) + '%.\n')
    else:
        print('The maximum efficiency is ' + str(round(optimal_efficiency, 3)) + '% with bandgaps of '
              + str(optimal_bandgaps[0:stack.number_of_bandgaps]) + ' eV.\n')
    # Give computation time.
    stop_time = time.time()
    print('More are given in file '+file_name+'. The calculations took')
    times = stop_time - start_time
    if times > 3600:
        print(str(round(times/3600, 3))+' hours.\n')
    if times > 60:
        print(str(round(times/60, 3))+' minutes.\n')
    else:
        print(str(round(times, 3))+' seconds.\n')
    return True    




def manage_input_output(
        file_name, T_cell, T_sun, spectrum, concentration, 
                   structure, composition, V_test, fc_rec_ratio, nonradiative_recombination_modeling, diffusion_limited, lifetimes, 
                   bulk_lifetimes, trap_lifetimes, SRVs, thicknesses, dopant_type, 
                   dopant_density, texturing, front_reflectance, rear_reflectance, acceptance_angle, 
                   MEG, max_yield, threshold_energy_normalized, sampling_range, bandgap_resolution, anything_variable):  
    """ Mostly delegates roles to other functions.
    Used by the module run.py."""
    
    print('\n') # Space out output
    start_time = time.time()  # Start timing calculations. 
    
    # Wrap up inputs into variables "stack" and "sampled_bandgaps."
    stack = Stack(T_cell, T_sun, spectrum, concentration, structure, composition,
                  V_test, fc_rec_ratio, nonradiative_recombination_modeling, diffusion_limited, lifetimes, bulk_lifetimes, trap_lifetimes, 
                  SRVs, thicknesses, dopant_type, dopant_density, 
                  texturing, front_reflectance, rear_reflectance, acceptance_angle, MEG, max_yield, 
                  threshold_energy_normalized, sampling_range, anything_variable)
    stack.counter==0# if you want to count something later
    stack.set_layer(stack, 0)
    sampled_bandgaps = SampledBandgaps(sampling_range, bandgap_resolution, spectrum)
    spectral.save_material_data(stack)  # Store materials' data like absorption coefficient and index of refraction
        
    # Calculate results
    file_data, optimal_bandgaps, optimal_efficiency, optimal_IV = stack_power.optimize_stack(sampled_bandgaps, stack)
    
    # Create outputs             
    make_plots(optimal_bandgaps, file_data, sampled_bandgaps, stack)    
    write_file(file_data, file_name, stack)
    display_results(start_time, optimal_efficiency, optimal_bandgaps, file_name, stack)
    
    return optimal_efficiency, optimal_IV
