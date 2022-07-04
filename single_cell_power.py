""" Created on Fri Aug 24 11:47:20 2018
Author: Nicholas Irvin

This module determines total power of multijunction cells and also saves data for each bandgap sampled."""


# Import libraries.
import numpy as np
from scipy.optimize import minimize_scalar
import single_cell_power, spectral, sunlight




def save_results(optimal_efficiency, optimal_IV, optimal_bandgaps, eff_all, IV_data, bandgaps, file_data, stack):
    """Store results to file_data and see if the new efficiency beats the previous optimum.
    Used by optimize_stack."""
    
    thickness = 'N/A' if(stack.thickness == 'N/A') else stack.thickness*1e4  # cm to um
    T_sun = stack.T_sun if stack.spectrum == 'Blackbody' else 'N/A'

    # Save results
    if stack.number_of_bandgaps != 1:
        file_data.append(bandgaps[0: stack.number_of_bandgaps] + [eff_all]
        + [stack.T_cell] + [stack.T_sun] + [stack.spectrum] + [stack.concentration] + [stack.structure] 
        + [stack.acceptance_angle] + [stack.texturing]  + [stack.rear_reflectance] 
        + [stack.composition] + [stack.nonradiative_recombination_modeling]
        + [thickness*1e4] + [stack.dopant_density] + [stack.dopant_type])        
    else:
        rec = IV_data.rec      
        file_data.append(bandgaps[:-1] + [eff_all] + [IV_data.Jsc/10]  # /10 to convert A/m^2 to mA/cm^2.
        + [IV_data.Voc] + [IV_data.FF] + [IV_data.Jmp/10] + [IV_data.Vmp] + [IV_data.Jph_mp/10]
        + [stack.V_test] + [IV_data.Jph_test/10] + ["{:.2e}".format(rec.dn)] + [rec.J_Auger/10] 
        + [rec.J_rad_front/10] + [rec.J_rad_back/10] + [rec.J_FCA/10] + [rec.J_trap/10] + [rec.J_SRV/10]
        + [rec.ERE] + [IV_data.PR] + [IV_data.Vdb]  
        + [stack.T_cell] + [T_sun] + [stack.spectrum] + [stack.concentration]
        + [stack.structure]  + [stack.acceptance_angle] + [stack.texturing] + [stack.rear_reflectance] 
        + [stack.composition] + [stack.nonradiative_recombination_modeling] + [rec.lifetime]
        + [rec.trap_lifetime] + [rec.radiative_lifetime] + [rec.Auger_lifetime] 
        + [stack.SRV] + [rec.diffusion_length*1e4] + [rec.electrical_diffusion_length*1e4] + [rec.ideal_electrical_diffusion_length*1e4] 
        + [rec.photon_recycling_diffusivity] + [rec.photon_recycling_L*1e4] + [IV_data.J_rad_abs/10] + [thickness]  # thickness converted to microns
        + ["{:.2e}".format(stack.dopant_density)] + [stack.dopant_type] + [stack.anything_variable] + [stack.anything_output])         
    # Update max efficiency and optimal bandgaps
    if (eff_all >= optimal_efficiency): 
        optimal_efficiency = eff_all;
        optimal_IV = IV_data;
        for i in range(stack.number_of_bandgaps):
            optimal_bandgaps[i] = bandgaps[i];
    return(optimal_efficiency, optimal_IV, optimal_bandgaps, file_data)



# if varying bandgaps
def independent_lookup_table(sampled_bandgaps, stack): 
    """ Purpose: Generate look-up table of maximum power (W m^-2) at every combination of 
    sampled_bandgaps.energies [E1, E2] specified by the sampled_bandgaps.lower_limits 
    and sampled_bandgaps.upper_limits vectors. The function stack can then use 
    this array to combine the power generate from each cell.
    
    The output is a matrix. Each array element is for a different point of [E1, E2] (eV). 
    Used by the function stack_efficiency. """
    
    array_size = len(sampled_bandgaps.energies)
    power_table = np.zeros((array_size, array_size)) # Power lookup 

    # Tack on infinity as the bandgap for space above the top cell.
    lower_limits = sampled_bandgaps.lower_limits + [sampled_bandgaps.infinity]
    upper_limits = sampled_bandgaps.upper_limits + [sampled_bandgaps.infinity]
    
    for i, E1 in enumerate(sampled_bandgaps.energies):
        for j, E2 in enumerate(sampled_bandgaps.energies):
            stack.set_bandgaps([E1, E2])
            # Check if need calculation.
            skip = 1
            for k in range(stack.number_of_bandgaps):
                if (E1 < E2 
                    and lower_limits[k] <= round(E1,4) <= upper_limits[k] 
                    and lower_limits[k+1] <= round(E2,4) <= upper_limits[k+1]):
                    skip = 0
            if skip == 0:
                power_table[i, j], IV_data  =  single_cell_power.max_power_pt(E1, E2, stack)

    return(power_table, IV_data)



# adds power of multiple junctions (often is also used for a single junction)
def independent_stack_power(bandgaps, sampled_bandgaps, stack):
    """ Purpose: Calculate total power (W m^-2) for a stack of cell wired without electrical constrictions between junctions. """
    power = 0  # Initialize
    for m in range(len(bandgaps)-1):  
        # Series connection, so add voltages while keeping current equivalent.
        stack.set_layer(stack, m)
        power_new, IV_data = single_cell_power.max_power_pt(bandgaps[m], bandgaps[m+1], stack)
        power = power + power_new
    return(power, IV_data)     
 
    
 

  
       
# for multijunctions
def series_stack_power(bandgaps, sampled_bandgaps, stack):
    """ Purpose: Calculate total power (W m^-2) for a stack of cell wired in series for a
    given set of bandgaps. Here we allow absortiance of the top cell to vary.
    Used by the function stack_efficiency. """

    # Vary absorptivities of upper cells for purpose of current matching.
    absorptivity_resolution = .05
    # An absorptivity_resolution of .1 means we will try each top layers at 
    # absorptivities of 0, .1, .2, .3, ... 1
    absorptivities = [1] + [absorptivity_resolution for x in range(stack.number_of_bandgaps-1)]
    power_max = 0

    for i in range(int((1/absorptivity_resolution)**len(bandgaps) + 1)): 
        # Find the stack's maximum current as the lowest Jsc.
        Jsc_array = []
        for j in range(len(bandgaps)-1): # - 1 because infinity counted in bandgaps
            stack.set_layer(stack, j)
            photocollection = spectral.Photocollection(bandgaps[j], stack)
            Jsc_incident = single_cell_power.find_current(0, bandgaps[j], bandgaps[j+1], photocollection, stack)
            Jsc_array = Jsc_array + [Jsc_incident]
            Jsc_low = min(Jsc_array)

        # Create total power function that is to be maximized by varying current.    
        def power(current):
            volt = 0  # Initialize
            for m in range(len(bandgaps)-1):  
                # Series connection, so add voltages while keeping current equivalent.
                stack.set_layer(stack, m)
                photocollection = spectral.Photocollection(bandgaps[m], stack)
                # need to reverify that absorptivities[m]*current works to help current matching
                volt = volt + single_cell_power.find_voltage(
                        absorptivities[m]*current, bandgaps[m], bandgaps[m+1], photocollection, stack, Jsc_array[m]) # photocollection[m], stack, Jsc_array[m])
            return -current*volt 
        res = minimize_scalar(power, bounds=[0, Jsc_low], method = 'Bounded')
        power_max = max(power_max, abs(res.fun))
        

        # Move to the next absorptivity configuration.
        if (stack.number_of_bandgaps == 1 
                or absorptivities >= [1 for i in range(stack.number_of_bandgaps)]):
             break  # Quit if it is the last configuration.
        cell = 1 #  Otherwise, start at the second cell.
        while (absorptivities[cell] >= 1):
            # For each cell above its absorptivity limits,
            absorptivities[cell] = absorptivity_resolution
            # return the absorptivity to its lower limit...
            cell += 1  # and move to the next cell.
        absorptivities[cell] = round(absorptivities[cell] + absorptivity_resolution, 5 )  # Increment the absorptivity...
        # And round to avoid numerical drifting

    return(power_max)     
 
    
    
    
 # for multijunctions, needs reverification if used
def parallel_stack_power(bandgaps, sampled_bandgaps, stack):
    """ Purpose: Calculate total power (W m^-2) for a stack of cell wired in parallel for a
    given set of bandgaps.
    Used by the function stack_efficiency. """ 

    if (bandgaps != sorted(bandgaps) or len(bandgaps) != len(set(bandgaps))): 
        return 0
    
    photocollections = [spectral.Photocollection(bandgaps[i], stack) for i in range(stack.number_of_bandgaps)]
    Jsc_array = [single_cell_power.find_Jsc(bandgaps[j], bandgaps[j+1], 0, photocollections[j], stack)
        for j in range(stack.number_of_bandgaps)]

    # Create total power function that is to be maximized by varying voltage. 
    def power(volt):
        current = np.zeros(stack.number_of_bandgaps)
        for k in range(stack.number_of_bandgaps):
            current[k] = single_cell_power.find_current(volt, bandgaps[k], bandgaps[k+1], photocollections[k], stack, Jsc_array[k])
        return -sum(current)*volt 
    res = minimize_scalar(power, bounds=[0, bandgaps[0]], method='Bounded')
    return abs(res.fun)   # Verify comparing to Saroj Pyakurel's report withoutarea decoupled



# for multijunctions, needs reverification if used
def combination_stack_power(bandgaps, sampled_bandgaps, stack):
    """N Silicon cells on bottom in parallel with other cells on top."""

    # Vary absorptivities of upper cells for purpose of current matching.
    absorptivity_resolution = .2
    # An absorptivity_resolution of .1 means we will try each top layers at 
    # absorptivities of 0, .1, .2, .3, ... 1
    absorptivities = [1] + [1] + [absorptivity_resolution for x in range(stack.number_of_bandgaps-2)]

    photocollections = [spectral.Photocollection(bandgaps[i], stack) for i in range(stack.number_of_bandgaps)]
    power_max = 0
    for i in range(int((1/absorptivity_resolution)**len(bandgaps) + 1)): 
        # Find the stack's maximum current as the lowest Jsc.
        power_N_max = 0
        Jsc_incident = [single_cell_power.find_Jsc(bandgaps[j], bandgaps[j+1], 0, photocollections[j], stack) for j in range(stack.number_of_bandgaps)] 
        Jsc_array = []
        for j in range(len(bandgaps)-1): # - 1 because infinity counted in bandgaps
            stack.set_layer(stack, j)
            Jsc_array = Jsc_array + [absorptivities[j]*Jsc_incident[j]
                        # Absorption of the light matched to the cell's bandgap... 
                        + sum(((1-absorptivities[m])/m)*Jsc_incident[m]    # + sum(((1-absorptivities[m])/m)*Jsc_incident[m]    ????
                        for m in range(j+1,stack.number_of_bandgaps))] 
                        # + Absorption of the light matched to higher cell's bandgap.
        Voc_bottom = single_cell_power.find_voltage(0, bandgaps[0], bandgaps[1], photocollections[0], stack)
        
        N_list = [i+1 for i in range(8)]            
        for N in N_list:         
            def power(current):
                """Just copy from intermediate band code"""
                current2 = current
                current3 = current  # Series condition
                voltage2 = single_cell_power.find_voltage(
                        current, bandgaps[1], bandgaps[1+1], photocollections[1], stack, Jsc_array[1])
                voltage3 = single_cell_power.find_voltage(
                        current, bandgaps[2], bandgaps[2+1], photocollections[2], stack, Jsc_array[2])
                voltage1 = voltage2 + voltage3  # Parallel condition
                if voltage1/N > Voc_bottom:
                    return(-(current2*voltage2 + current3*voltage3))
    
                current1 = single_cell_power.find_current(
                        voltage1/N, bandgaps[0], bandgaps[0+1], photocollections[0], stack, Jsc_array[0])/N
                return -(current1*voltage1 + current2*voltage2 + current3*voltage3)
            res = minimize_scalar(power, bounds=[0, min(Jsc_array[1:])], method = 'Bounded')
            
            power = -res.fun
            if power > power_max:
                power_max = power
            if power > power_N_max:
                power_N_max = power
                # p_rint(N, power/10)

        # Move to the next absorptivity configuration.
        if (stack.number_of_bandgaps == 1 
                or absorptivities >= [1 for i in range(stack.number_of_bandgaps)]):
             break  # Quit if it is the last configuration.
        cell = 2 #  Otherwise, start at the third cell --- because don't need silicon to current match
        while (absorptivities[cell] >= 1):
            # For each cell above its absorptivity limits,
            absorptivities[cell] = absorptivity_resolution
            # return the absorptivity to its lower limit...
            cell += 1  # and move to the next cell.
        absorptivities[cell] = round(absorptivities[cell] + absorptivity_resolution, 5 )  # Increment the absorptivity...
        # And round to avoid numerical drifting
    return(power_max)     



 
# special type of cell, needs reverification if used
def intermediate_band_power(bandgaps, sampled_bandgaps, stack):    
    """ Purpose: Calculate total power (W m^-2) for a intermediate-band cell. This is
    modeled by a bottom and middle cell in series followed by connection to the 
    top cell in parallel.
    Used by the function stack_efficiency. """ 
    
    # The bottom cells' bandgaps must add up to the top cell bandgap.
    if (bandgaps[0] >= bandgaps[1]) or not np.isclose(bandgaps[0] + bandgaps[1], bandgaps[2]):
        return 0
    
    photocollections = [spectral.Photocollection(bandgaps[i], stack) for i in range(stack.number_of_bandgaps)]
    Jsc_array = [single_cell_power.find_Jsc(bandgaps[j], bandgaps[j+1], 0, photocollections[j], stack) for j in range(stack.number_of_bandgaps)]
    Jsc_eff = min(Jsc_array[0], Jsc_array[1]) 
    # The current through the two in series is limited by the lowest Jsc.
        
    if stack.number_of_bandgaps != 2:
        return False
    if stack.number_of_bandgaps == 3:
            # Create total power function       that is to be maximized by varying current 
            # through the intermediate band.
            def power(current):
                current1 = current
                current2 = current  # Series condition
                voltage1 = single_cell_power.find_voltage(
                        current, bandgaps[0], bandgaps[0+1], photocollections[0], stack, Jsc_array[0])
                voltage2 = single_cell_power.find_voltage(
                        current, bandgaps[1], bandgaps[1+1], photocollections[1], stack, Jsc_array[1])
                voltage3 = voltage1 + voltage2  # Parallel condition
                current3 = single_cell_power.find_current(
                        voltage3, bandgaps[2], bandgaps[2+1], photocollections[2], stack, Jsc_array[2]) 
                return -(current1*voltage1 + current2*voltage2 + current3*voltage3)
            return(abs(minimize_scalar(power, bounds=[0, Jsc_eff], method='Bounded').fun))  # Output max power.




def stack_efficiency(bandgaps, stack, sampled_bandgaps, power_table=None, IV_data=None):
    """This function controls how the stack's efficiency (%) is calculated based on
    the solar structure.
    Used by optimize_stack."""
    
    stack.set_bandgap(bandgaps[0]) 
    power_in = sunlight.Incident_Light(stack.spectrum, stack).irradiance
    
    # Calculate electrical power generated    
    if stack.structure == 'Independent' and stack.composition == [] and stack.number_of_bandgaps > 1:  # Independent connection
        eff_all = 0.0 
        for i in range(stack.number_of_bandgaps):  # For each cell,
            # find the elements in the power lookup table that correspond to... 
            index_bottom = np.where(np.isclose(sampled_bandgaps.energies, bandgaps[i]))[0][0]
            # the cell's lower ..
            index_top = np.where(np.isclose(sampled_bandgaps.energies, bandgaps[i+1]))[0][0]
            # and upper energy limits...
            eff_all  =  eff_all + 1e2*power_table[index_bottom, index_top]/power_in
            # to find output power between those limits.
    else: # (stack.number_of_bandgaps > 1)
        if stack.structure == 'Independent' and stack.composition == []:
            power = 0
            for m in range(stack.number_of_bandgaps): 
                stack.set_layer(stack, m)
                power_new, IV_data = single_cell_power.max_power_pt(bandgaps[m], bandgaps[m+1], stack)
                power = power + power_new
        elif stack.structure == 'Independent' and stack.composition != []:
                power, IV_data = independent_stack_power(bandgaps, sampled_bandgaps, stack)           
        elif stack.structure == 'Series':                
                power, IV_data = [series_stack_power(bandgaps, sampled_bandgaps, stack), 'N/A']
        elif stack.structure == 'Parallel':                
                power, IV_data = [parallel_stack_power(bandgaps, sampled_bandgaps, stack), 'N/A']
        elif stack.structure == 'Combination':    
                power, IV_data = [combination_stack_power(bandgaps, sampled_bandgaps, stack), 'N/A']        
        elif stack.structure == 'Intermediate':           
                power, IV_data = [intermediate_band_power(bandgaps, sampled_bandgaps, stack), 'N/A']
        eff_all = 100*power/power_in                
    return(eff_all, IV_data)  # Efficiency likely accurate to three decimal places - dependent on number of integral subintervals
    
    
    
    

def optimize_stack(sampled_bandgaps, stack, power_table=None, IV_data=None):
    """ Purpose: Maximize the efficiency (%) such that, for each cell, E1 < E2, 
    and the E2 of a cell must be the E1 of the cell above it. 
    Used by module run. """
            
    # Create the list that stores the bandgap values that will be iteratively sampled.  
    bandgaps = sampled_bandgaps.lower_limits + [sampled_bandgaps.infinity]
    # Initialize Results.
    optimal_bandgaps = np.ones(stack.number_of_bandgaps) # Optimum band gaps, 
    optimal_efficiency = 0.0; # Max efficiency
    optimal_IV = None; # Max efficiency
    file_data=[]

    # For independent connections, create a lookup table to cut computation time
    if stack.structure == 'Independent' and stack.composition == [] and stack.number_of_bandgaps > 1:
        power_table, IV_data = independent_lookup_table(sampled_bandgaps, stack)
                        
    # For each bandgap configuration...
    for j in range(len(sampled_bandgaps.energies)**stack.number_of_bandgaps):
        stack.set_bandgaps(bandgaps)
        # Calculate the stack's efficiency.
        eff_all, IV_data = stack_efficiency(bandgaps, stack, sampled_bandgaps, power_table, IV_data)
        # Save?
        # if eff_all != 0:
        optimal_efficiency, optimal_IV, optimal_bandgaps, file_data = save_results(
                optimal_efficiency, optimal_IV, optimal_bandgaps, eff_all, IV_data, 
                bandgaps, file_data, stack)   
        
        # Move to the next bandgap configuration.
        if bandgaps >= sampled_bandgaps.upper_limits:
             break  # Quit if it is the last configuration.
        cell = 0 #  Otherwise, start at the bottom cell.
        while ((bandgaps[cell] >= sampled_bandgaps.upper_limits[cell]) 
                or (bandgaps[cell] >= (bandgaps[cell+1] - sampled_bandgaps.bandgap_resolution))):
            # For each cell above its upper limits,
            bandgaps[cell] = sampled_bandgaps.lower_limits[cell] # return the bandgap to its lower limit...
            cell += 1  # and move to the next cell.
        bandgaps[cell] = round( bandgaps[cell] + sampled_bandgaps.bandgap_resolution, 5 )  # Increment the bandgap...
     
    return(file_data, optimal_bandgaps, optimal_efficiency, optimal_IV)
