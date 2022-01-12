""" Created on Tue Nov 27 16:21:24 2018
Author: Nicholas Irvin

Outline: This module is used for spectral quantities - quantities that vary with energy. 
These energy-dependent quantities items are made into arrays like 'n_per_eV'. 
A separate array, such as'eVs' or 'E', stores the energy of each element.       
Important outputs are absorptance, EQE, and photoflux (radiative recombination)."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapz
import sunlight, carrier_models
plt.rc('font', family='arial', size = '16')

c = 2.99792458e8               
# m/s  The speed of light
h = 6.6260755e-34              
# J*s  Planck's constant
k = 1.380658e-23               
# J/K  Boltzmann's constant
q = 1.60217733e-19             
# C    The charge of an electron
g = 2*np.pi/(c**2)/(h**3)      
# s^5/(kg^3*m^8)  The coefficient for particle flux
f = 1/46260 
# (sr)  (The solid angle of the sun from Earth)/Pi
inf = float('inf')   


def find_bandgaps(composition):  
    """ Give as list like ['GaAs'] or ['Si', 'GaPN'] 
    outputs bandgap in eVs in the form that input_ouput_management.SampledBandgaps needs: [[Bandgap1, Bandgap1], [Bandgap2, Bandgap2], etc] 
    i.e. a list of lists.
    Used by the main run function."""
    # this number isn't actually critical becauses we just use the absorption coefficient data anyway.
    bandgap_dict = {'GaAs': 1.423, 'Si': 1.125, 'CIGS': 1.115, 'CIS': 1.016, 'CdTe': 1.488, 'CdTe monocrystalline': 1.514, 'GaNP': 1.96, 'perovskite triple cation': 1.59, 'perovskite MaPI': 1.58}  # Double check the GaPN
    # Common bandgaps: Si 1.125 so use sampling_range = [[1.125, 1.125]]. CdTe 1.514, CIS 1.016, CIGS 1.115, https://aip-scitation-org.ezproxy1.lib.asu.edu/doi/pdf/10.1063/1.4767120?class=pdf
    # GaAs 1.423 Temperature dependence of semiconductor band gaps by O'Donnell    
    sampling_range = [[bandgap_dict[comp], bandgap_dict[comp]] for comp in composition]
    return(sampling_range)      

def save_material_data(stack): 
    """ Saves data to the main data object 'stack'.
    Used by input_output_management.manage_input_output """
    stack.eVs_0 = sunlight.AM_spectra('AM1.5G', stack).energies  # store AM1.5G energy values to calculate things at
    # eVs range from 0.309960612 to 4.428008739
    
    # import abosorption coefficent from Excel
    if stack.composition != []:
        alpha_list = []
        nr_list = []
        for comp in stack.composition:         
            df_a = pd.read_csv(r'Tabulated_Parameters\absorption_coefficient.csv')
            alpha = df_a[comp].values  # and array resolved by energy
            # alpha = df_a['GaAs_Sturge'].values  # uncomment for Miller (2012)'s absorption coefficient
            alpha_list += [alpha]  # a list of arrays
            df_n = pd.read_csv(r'Tabulated_Parameters\index_of_refraction.csv')
            nr = df_n[comp].values 
            nr_list += [nr]
            if comp == 'CdTe':
                df_a = pd.read_csv(r'Tabulated_Parameters\absorption_coefficient.csv')
                stack.sigma_FCA = df_a['CdTe_FCA'].values  # and array resolved by energy
        stack.alpha_list = alpha_list
        stack.nr_list = nr_list


        
        
        

def blackbody(volt, E1, E2, stack, n=5*10**3, eVs=[]):  # 5e3
    """
    Output a vector of blackbody_photocurrent in (s^-1 m^-2 eV^-1)
    along with the energy vector (E) that it is resolved at. 
    """
    n = int(n)
    if eVs != []:
        E = q*eVs  # Use given energy array, convert from (eV) to (J)
    else:        # make the energy array
        # n = 5*10**3  # (default) Number of integrals subintervals usually 5*10**3 or 10**4
        # QY is quantum yield (seperate from quantum efficiency), should be 1 unless special case of multiple exciton generation.
        w = (E2-E1)/(n-1) # Subinterval width from interval E2-E1. Multiply by q to convert from eV to J.
        if stack.layer_num == 0 and stack.composition != []:  # start integrating below bandgap for bottom cell in multijunction
            E1 = 0
        E = q*np.linspace(E1 + w/2, E2 - w/2, n)  # Vectorize the energy space.   
    QY = find_QY(E/q, stack.bandgap, stack)  # Quantum yield for MEG calculations
    # blackbody_photocurrent =  (QY*g*E**2  /abs(  np.exp((E-QY*q*volt)/(k*stack.T_cell)) - 1))  # spectral photocurrent (s^-1 m^-2 eV^-1) with Bose-Einstein statistics, leads to singularities so can be bad for integral
    blackbody_photocurrent =  (QY*g*E**2  *  np.exp(-E/(k*stack.T_cell)) * np.exp((QY*q*volt)/(k*stack.T_cell)) )   # spectral photocurrent (s^-1 m^-2 eV^-1) with Maxwell-Boltzmann approximation
    return [E, blackbody_photocurrent]




  
 
class Absorption:
    def __init__(self, Eg, stack, volt=0, tau=1e11, diffusion_length=1e11, diffusivity=1e11, rec='N/A', eVs=0):
        """ Compile absorptance, rear_emittance, index of refraction, and absorption coefficient
        at each energy in eVs as well as the carrier concentration. """        

        eVs = stack.eVs_0   # Energies to calculate data at   
        # change energies to integrate over - important for multijunctions
        E1_index = np.searchsorted(eVs, Eg, side='left')
        if stack.layer_num == 0:  # start integrating below bandgap for bottom cell in multijunction
            E1_index = 0
        E2_index = np.searchsorted(eVs, stack.bandgaps[stack.layer_num+1])  # in multijunctions, E2 is used to seperate cells' spectra
        eVs = eVs[E1_index:E2_index+1]
        stack.n_per_eV = sunlight.Incident_Light(stack.spectrum, stack).n_per_eV[E1_index:E2_index+1]  # Incident sunlight, spectral photon flux for in 1/s*m^2*eV:
        stack.eVs = eVs
        
        stack.bandgap_index = np.searchsorted(eVs, stack.bandgap)
        Rf_ext = stack.front_reflectance  # Front external reflectance (reduce with ARC)
        Rb = stack.rear_reflectance       # Rear reflectance. If not than bifacial than want to increase.
        theta = stack.acceptance_angle

        if stack.composition == []:  # Take absorptance/EQE as a step function
            absorptance = (1-Rf_ext)*np.array([0 if(e<Eg) else 1 for e in eVs])
            alpha = np.array([0 if(e<Eg) else 10**8 for e in eVs])
            rear_emittance = np.zeros(len(eVs))  # assume no rear emission
            nr = np.ones(len(eVs))
            alpha_FCA = 0
            Rf_int = (Rf_ext+(nr/np.sin(theta))**2-1)*(np.sin(theta)/nr)**2  # reflectance increased by total internal reflection

        else:  # Calculate absorptance/EQE from absorption coefficient
            W = stack.thickness
            alpha = stack.alpha_list[stack.layer_num]  # data saved in the function save_material_data 
            alpha = alpha[E1_index:E2_index+1]
            nr = stack.nr_list[stack.layer_num]   
            nr = nr[E1_index:E2_index+1]
            Rf_int = (Rf_ext+(nr/np.sin(theta))**2-1)*(np.sin(theta)/nr)**2 # no acceptance angle then (Rf_ext+nr**2-1)/nr**2 #  Does this change Martin's Lt? Rf_ext*(1-(np.sin(stack.acceptance_angle)/nr)**2) # Front internal reflectance found by averaging reflectance of Lambertian distribution with piecewise reflectance: Rf_ext before critical angle and 1 after critical angle (assuming total internal refection)           

            # Consider free carrier absorption?
            try:
                carriers = carrier_models.Carriers(volt, stack)
                go = 1
            except:  # no carriers model so no FCA
                alpha_FCA = np.zeros(len(eVs))
                go = 0
            if go == 1:
                if stack.composition[stack.layer_num] == 'Si':  # "Parameterization of free carrier absorption in highly doped silicon for solar cells" 
                    carriers = carrier_models.Carriers(volt, stack)
                    alpha_FCA = 2.6e-18*(h*c/(eVs*q)*1e6)**2.4*carriers.p + 1.8e-18*(h*c/(eVs*q)*1e6)**2.6*carriers.n
               
                elif stack.composition[stack.layer_num] == 'GaAs':  
                    carriers = carrier_models.Carriers(volt, stack)
                    alpha_FCA = 8.3e-18*(h*c/(eVs*q)*1e6)**2*carriers.p + 4e-29*(h*c/(eVs*q)*1e9)**3*carriers.n  
                                                                # p-type from  Free-carrier absorption in Be- and C-doped GaAs epilayers and far infrared detector applications     
                                                                # n-type from Clugston citing the source J. I. Pankove, Optical Processes in Semiconductors, pp. 75±76, Dover Publications, New York, 1971.                                                                                                                
                elif stack.composition[stack.layer_num] == 'CdTe':  
                    carriers = carrier_models.Carriers(volt, stack)
                    sigma_e = stack.sigma_FCA  # n-type from FREE CARRIER ABSORPTION IN n-TYPE CdTe
                    sigma_h = 2.1e-17/1.4e-19*sigma_e  # inspired by "Excitation-dependent carrier lifetime and diffusion length in bulk CdTe determined by time-resolved optical pump-probe techniques "
                    alpha_FCA = carriers.n*sigma_e + carriers.p*sigma_h                  
                elif stack.composition[stack.layer_num] == 'perovskite triple cation':           
                    sigma = 1.5e-17*(h*c/(eVs*q)*1e6)**2.5  # this is taken from 1053 nm from "A carrier density dependent diffusion coefficient, recombination rate and diffusion length in MAPbI3 and MAPbBr3 crystals measured under one- and two-photon excitations"
                    alpha_FCA = (carriers.n+carriers.p)*sigma  # free carrier absorption cross section values, seh = 1.6e17 cm^-3
                else:  # no FCA model so no FCA
                    alpha_FCA = np.zeros(len(eVs)) + 1e-16               # (1/cm)        
            alpha_t = alpha + alpha_FCA + 1e-60 # Total absorption is useful + useless + 1e-16 to avoid divide by 0 
            

            # must recalculate absorptance at each voltage because of dependance on FCA
            if stack.texturing == 'No':
                absorptance = ((1-Rf_ext)*alpha/alpha_t*  #  *alpha/alpha_t because FCA isn't radiating
                    (((1-np.exp(-alpha_t*W))*(1+Rb*np.exp(-alpha_t*W)))  # this line = planar case
                    /(1-Rb*Rf_ext*np.exp(-2*alpha_t*W))))  # Note that Rf_ext = 0 ideally so the denominator is 1 ideally!!                
            elif stack.texturing == 'Yes':
                # absorptance = 4*nr**2*alpha*W / (4*nr**2*(alpha+alpha_FCA)*W + 1)  # Yablonovitch Limit
                # absorptance = alpha/(alpha_t + 1/(50*W))  # For silicon, used by Richter
                # absorptance = ((1-np.exp(-4*alpha*W))/(1-np.exp(-4*alpha*W)*(1-nr**-2)))  # from Holman
                ## Real absorptance found below with Eq. 5 from Martin's 2002 "Lambertian Light Trapping" with Schafer's alpha/alpha_tot
                x = 0.935*(alpha_t*W)**0.67
                LT = (2+x)/(1+x)*W # Average path length photon travels from front to back of cell.
                    # Richter and many others assumed LT = W
                absorptance = ((1-Rf_ext)*alpha/alpha_t*  #  *alpha/alpha_t because FCA isn't radiating
                (((1-np.exp(-alpha_t*LT))*(1+Rb*np.exp(-alpha_t*LT)))
                /(1-Rb*Rf_int*np.exp(-2*alpha_t*LT))))                                  
           
            # Rear emittance calculation:
            if np.all(Rb==1):  # If reflectance is perfect, no rear emission
                rear_emittance = np.zeros(len(eVs))
            else:  # Calculate rear_emittance
                if stack.texturing == 'No' : # USe a faster version of Eq 12 from Ganapati's 2016 Ultra-Efficient Thermophotovoltaics
                    theta_c = np.arcsin(1/nr)  # critical angle
                    bound1 = 2*(1-np.cos(theta_c))*1/np.sin(theta_c)**2
                    bound2 = 2*1/np.cos(theta_c)  
                    x = 1.88*(alpha_t*W)**0.59  # My own parametrization, optimized to 0.1 um GaAs with Rb = 50%.
                    alpha_LT_in = alpha_t*(bound1+x)/(1+x)  # alpha* average path length factor for photons in escape cone  #  *alpha/alpha_t because FCA isn't radiating
                    alpha_LT_out = alpha_t*1/np.cos(theta_c) * (np.cos(theta_c)*bound2+x)/(1+x)  # alpha* average path length factor for photons out of escape cone

                    emittance_in = ((1-Rb)*(1-np.exp(-alpha_LT_in*W))*(1+Rf_ext*np.exp(-alpha_LT_in*W))                                 
                                    /(1-Rb*Rf_ext*np.exp(-2*alpha_LT_in*W)))
                    emittance_out =  ((1-Rb)*(1-np.exp(-2*alpha_LT_out*W))
                            /(1-Rb*np.exp(-2*alpha_LT_out*W)))
                    rear_emittance = ( 1/nr**2*emittance_in + (1-1/nr**2)*emittance_out )*alpha/alpha_t  #  *alpha/alpha_t because FCA isn't radiating
                    # includes parasitic FCA. If want bifacial rear_QE, multiply by alpha/alpha_t

              
                if stack.texturing == 'Yes':  # From Martin's 2002 paper again
                    x = 0.935*(alpha_t*W)**0.67
                    LT = (2+x)/(1+x)*W # Average path length W from front to back
                    rear_emittance = (1-Rb)*(((1-np.exp(-alpha_t*LT))*(1+Rf_int*np.exp(-alpha_t*LT))) *alpha/alpha_t
                        /(1-Rb*Rf_int*np.exp(-2*alpha_t*LT)))                              
                   

            # ## Can plot absorptance or emittance 
            # if volt == stack.V_test:
            #     fig, ax = plt.subplots()
            #     plt.title(stack.texturing)
            #     plt.xlabel('Energies (eVs)')
            #     plt.ylabel('absorptance (%)')
            #     ax.tick_params(which='both', direction='in', right='True', top='true')
            #     plt.plot(eVs, 100*absorptance, label='abs')
            #     plt.legend(fontsize='small')      
                
        self.absorptance = absorptance
        self.rear_emittance = rear_emittance
        self.nr = nr
        self.alpha = alpha
        self.alpha_FCA = alpha_FCA
        self.Rf_int = Rf_int
        self.Rf_ext = Rf_ext
        stack.absorption = self
        def scale_absorptance(self, scalar):
            # 0 <= scalar <= 1
            self.absorptance = scalar*self.absorptance




class Photocollection:  # gets EQE
    def __init__(self, Eg, stack, volt=0, tau=1e11, diffusion_length=1e11, diffusivity=1e11, absorptance=0, rear_emittance=0, rec='N/A', eVs=0):  # cut number of optional inputs
        """ Compile absorptance, rear_emittance, EQE, rear_QE, index of refraction, and absorption coefficient
        at each energy in eVs as well as the carrier concentration. """        

        absorption = Absorption(Eg, stack, volt, tau, diffusion_length, diffusivity, rec, eVs)
        absorptance = absorption.absorptance
        rear_emittance = absorption.rear_emittance
        nr = absorption.nr
        alpha = absorption.alpha
        alpha_FCA = absorption.alpha_FCA
        Rf_int = absorption.Rf_int
        Rf_ext = absorption.Rf_ext
        n_per_eV = np.array(sunlight.Incident_Light(stack.spectrum, stack).n_per_eV)  # move out?               
        eVs = stack.eVs
        n_per_eV = stack.n_per_eV        
        
        W = stack.thickness
        Rb = stack.rear_reflectance       # Rear reflectance. If not than bifacial than want to increase.
        alpha_t = alpha + alpha_FCA
            
        """ Calculate EQE with multi-pass model: "Silicon solar cells reaching the efficiency limits: from simple to complex modelling" """    
        SRV = stack.SRV
        L = diffusion_length
        D = diffusivity 
        # Sometimes don't need the diffusion equation:
        if stack.diffusion_limited == 'No' or L==1e11:
            EQE = absorptance  # collection_efficiency should be 1 so bypass calculation
            rear_QE = rear_emittance
            
            QY = find_QY(eVs, Eg, stack)  # Multi-exciton-generation (MEG). # MEG refers to when Quantum Yield (QY) > 1.
            spectral_flux = EQE*n_per_eV*QY
            J = q*trapz(spectral_flux, eVs)

            
        elif stack.diffusion_limited == 'Yes':  # EQE limited, so calculate EQE
            normalization_fact = n_per_eV*absorptance/np.trapz(n_per_eV*absorptance, eVs)  # Normalization used to split up injection over energy spectrum evenly
            try:
                carriers = rec.carriers
            except:
                carriers = carrier_models.Carriers(volt, stack)
            n_maj = carriers.p if stack.dopant_type == 'p'  else carriers.n

            """ Get EQE """
            def get_EQE(L, tau, SRV):#, E=0):  
                # Boundaries chosen: dn(0) = 0, Sn(W) > 0
                # Hovel model extended beyond single pass to double pass and Lambertian generation function
                # Substituded 0 for Ze and Wb, and W for ZB for Eq. A3-A5 in "Silicon solar cells reaching the efficiency limits: from simple to complex modelling". Code at https://github.com/kowalczewski/Lisa                   
                #  Discrepency between my EQE and PC1D EQE only between 1.15 eV (generation flat) and 1.7 eV (generation = alpha not (50 alpha). Which means its the light-trapping or diffusion model different.                    # -> See which one Silvaco matches.
                gamma = L **2 / (D * (1-alpha_LT**2*L**2 )) 
                S = SRV 
                A_dark = carriers.ni_eff**2/n_maj* (np.exp(q*volt/(k*stack.T_cell)) - 1)*normalization_fact *1e4  # 1e4 for 1/cm^2 to 1/m^2                         
                cosh = np.cosh(W/L) if W/L<100 else 1e43
                sinh = np.sinh(W/L) if W/L<100 else 1e43
                B_dark = (-A_dark* (D/L*sinh + S*cosh)
                      / (D/L*cosh + S*sinh))                    
                J_dark = q*D*(B_dark/L) 
                
                A = (carriers.ni_eff**2/n_maj* (np.exp(q*volt/(k*stack.T_cell)) - 1)*normalization_fact *1e4  # (m^-1 s^-1)  Use 1e4 for 1/cm^2 to 1/m^2                       
                     - gamma* (a_minus + a_minus*Rb*np.exp(-2*alpha_LT*W)  ))  
                B = (( -A* (D/L*sinh + S*cosh)  # B also units of m^-1 s^-1
                      -gamma*(a_minus*np.exp(-alpha_LT*W)*(S - alpha_LT*D) 
                              + a_minus*Rb*np.exp(-alpha_LT*W)  # a_plus = a_minus*Rb*np.exp(-2 alpha_LT*W)
                              *(S+alpha_LT*D)))
                     / (D/L*cosh + S*sinh))                    
                J = q*D*(B/L + gamma*(-alpha_LT*a_minus 
                                        + alpha_LT*a_minus*Rb*np.exp(-2*alpha_LT*W)))  # ()     # limit as L-> inf, dn->0 is q*absortpance*(1-S/alpha/D)?
                EQE = (J-J_dark)/(q*n_per_eV+1e-32)
                J = np.trapz(J, eVs)  
                Z = np.linspace(0, W, 101) 
                stack.Z = Z
                if volt == 0:
                    stack.dn_z = np.ones(len(Z))
                else:
                    cosh_z = [np.cosh(z/L) if z/L<20 else 1e10 for z in Z]
                    sinh_z = [np.sinh(z/L) if z/L<20 else 1e10 for z in Z] # 1e43
                    dn_z = trapz([(A*cosh_z[i] + B*sinh_z[i]
                      + gamma*(a_minus*np.exp(-alpha_LT*z)
                              + a_minus*Rb*np.exp(-alpha_LT*(2*W-z))))*1e-4  # (1/cm^3)
                            for i, z in enumerate(Z)], eVs)
                    stack.dn_z = np.abs(dn_z) 
    
                return J, EQE 
            
            if stack.texturing == 'No':
                alpha_LT = alpha_t    
                a_minus = (((1-Rf_ext)*alpha_LT)*alpha/alpha_t*n_per_eV  # (cm^-1) #  *alpha/alpha_t because FCA isn't radiating
                            /(1 - Rb*Rf_ext*np.exp(-2*alpha_t*W)))  # Note that Rf_ext = 0 ideally so the denominator is 1 ideally!!
            elif stack.texturing == 'Yes':
                x = 0.935*(alpha_t*W)**0.67

                alpha_LT =  (2+x)/(1+x)*alpha_t # Generative absorption times average path length/thickness       
                # Add parasitic back in? And for rear?                                  
                a_minus = ((1-Rf_ext)*alpha_LT*alpha/alpha_t*n_per_eV
                           / (1 - Rb*Rf_int*np.exp(-2*alpha_LT*W)))   # (cm^-1) 
            J, EQE = get_EQE(L, tau, SRV)
            
            """ rear_QE for radiative recombination out the rear"""
            # Substituded 0 for Ze and Wb, and W for ZB for Eq. A3-A5 in "Silicon solar cells reaching the efficiency limits: from simple to complex modelling
            if np.all(Rb==1):  # If reflectance isn't perfect, calculate radiative emission loss to backmirror
                rear_QE = np.zeros(len(eVs))
            else:

                def get_rear_QE(L, tau, SRV, alpha_LT, a_minus, Rf):  
                    rear_normalization_fact = n_per_eV*rear_emittance/np.trapz(n_per_eV*rear_emittance, eVs)  # Normalization used to split up injection over energy spectrum evenly
                    gamma = L **2 / (D * (1-alpha_LT**2*L**2 ))                                            
                    # Hovel model extended beyond single pass to double pass and Lambertian generation function
                    # Ze = W, We = 0 = Z for Eq. A9-A11 in "Silicon solar cells reaching the efficiency limits: from simple to complex modelling
                    # We use the emitter equations for the rear_QE of the base as it effectively flips the sunlight source to the other side.                              
                    A_dark = carriers.ni_eff**2/n_maj* (np.exp(q*volt/(k*stack.T_cell)) - 1)*rear_normalization_fact*1e4  # normalization to go to units of per_eV, *1e4 to go from 1/cm^2 to 1/m^2
                    cosh = np.cosh(W/L) if W/L<100 else 1e43
                    sinh = np.sinh(W/L) if W/L<100 else 1e43
                    B_dark = ((-A_dark* (D/L*sinh + SRV*cosh))  
                          / (D/L*cosh + SRV*sinh))
                    J_dark = -q*D*( -B_dark/L)             

                    A = (carriers.ni_eff**2/n_maj* (np.exp(q*volt/(k*stack.T_cell)) - 1)*rear_normalization_fact*1e4  # normalization to go to units of per_eV, *1e4 to go from 1/cm^2 to 1/m^2
                        - gamma*(a_minus*np.exp(-alpha_LT*W) + 
                                  a_minus*Rf*np.exp(-alpha_LT*W)))  # a_plus = a_minus*Rf*np.exp(-2 alpha_LT*W)
                    B = (( -A* (D/L*sinh + SRV*cosh)  
                          -gamma*( a_minus*(SRV + alpha_LT*D) 
                                  + a_minus*Rf*np.exp(-2*alpha_LT*W)*(SRV - alpha_LT*D)))
                          / (D/L*cosh + SRV*sinh))
                    J = -q*D*( -B/L + gamma*( -alpha_LT*a_minus*np.exp(-alpha_LT*W) 
                        + alpha_LT*a_minus*Rf*np.exp(-alpha_LT*W) ))  # An array, will integrate
                    rearQE = (J-J_dark)/(q*n_per_eV+1e-32)
                    return rearQE

                if stack.texturing == 'No': 
                    theta_c = np.arcsin(1/nr)  # critical angle
                    bound1 = 2*(1-np.cos(theta_c))*1/np.sin(theta_c)**2
                    """ still work if simlpify bound2??"""
                    bound2 = 2*1/np.cos(theta_c)     
                    x = 1.88*(alpha*W)**0.59 
                    alpha_LT_in = (bound1+x)/(1+x)*alpha_t  # alpha* average path length factor for photons in escape cone
                    alpha_LT_out = 1/np.cos(theta_c)*(np.cos(theta_c)*bound2+x)/(1+x)*alpha_t  # alpha scaled by average path length factor for photons out of escape cone
                    # Hypothetical flux from rear substrate 
                    a_minus_in = (((1-Rb)*alpha_LT_in)*alpha/alpha_t*n_per_eV #  alpha/alpha_t because FCA isn't radiating
                                    / (1 - Rb*Rf_ext*np.exp(-2*alpha_LT_in*W)))
                    a_minus_out = (((1-Rb)*alpha_LT_out)*alpha/alpha_t*n_per_eV
                                    / (1 - Rb*np.exp(-2*alpha_LT_out*W)))

                    rear_QE = ((1/nr**2)*get_rear_QE(L, D, SRV, alpha_LT_in, a_minus_in, Rf_ext)  # 1/nr**2 weights by number of photons in escape cone
                           + (1-1/nr**2)*get_rear_QE(L, D, SRV, alpha_LT_out, a_minus_out, 1))  # weight by number out of escape cone
                elif stack.texturing == 'Yes':
                    x = 0.935*(alpha*W)**0.67
                    alpha_LT =  (2+x)/(1+x)*alpha_t # Averaged path length factor                                     
                    a_minus = ((1-Rb)*alpha_LT*alpha/alpha_t*n_per_eV
                               / (1 - Rb*Rf_int*np.exp(-2*alpha_LT*W)))   # (cm^-1)  
                    rear_QE = get_rear_QE(L, tau, SRV, alpha_LT, a_minus, Rf_int)

                    
            # # can save EQE data to Excel
            # import csv            
            # file_name = 'export.csv'
            # file = open(file_name, 'w+', newline='')
            # # import xlsxwriter
            # # writer = csv.writer(file)
            # # writer.writerow(eVs) 
            # # writer.writerow(EQE)   
            # # file.close()

            # ## Can plot EQE
            # if volt == stack.V_test:
            #     fig, ax = plt.subplots()
            #     plt.title(stack.texturing)
            #     plt.xlabel('Energies (eVs)')
            #     plt.ylabel('EQE (%)')
            #     ax.tick_params(which='both', direction='in', right='True', top='true')
            #     plt.plot(eVs, 100*EQE, label='abs')
            #     plt.legend(fontsize='small')   

                
        self.absorptance = absorptance
        self.EQE = EQE
        self.J = J
        self.rear_emittance = rear_emittance
        self.rear_QE = rear_QE
        self.nr = nr
        self.alpha = alpha
        self.alpha_FCA = alpha_FCA
        self.Rf_int = Rf_int
        self.Rf_ext = Rf_ext
        stack.Current_equations = self

    
    
    
    
def find_QY(eVs, Eg, stack):  
    # Quantum yields for special multi-exciton cells, usually ignore
    
        if stack.MEG == 'No':
            return np.ones(len(eVs))
        
        # An imposed bound on quantum yield:
        max_yield = stack.max_yield  # For a practically infinite max, use 10**4.
        energy_threshold_normalized = stack.energy_threshold_normalized
        # Minimum photon energy required for QY > 1: 
        energy_threshold = energy_threshold_normalized*Eg 
        # Ideal energy_threshold is 2Eg and record cells have 2.2Eg.
        QY_slope = 1/(energy_threshold-Eg)
            
        # Calculate quantum yield, the ratio of excited electrons to photons at each energy in eVs array.
        if energy_threshold_normalized == 2:  # Ideal quantum yield is a step-function.
            QY = np.minimum(eVs//Eg, max_yield)
        else:  # Non-ideal Quantum yield is modelled as flat then linearly increasing.
            QY = np.array([min(
                    QY_slope*(eV-energy_threshold) + 1 if(eV>energy_threshold) else 1
                    , max_yield) for eV in eVs])
            # This linear model actually achieves higher efficiencies than the
            # staircase model under full concentration. See Hanna, Nozik's "Effect of Solar Concentration" 2012.
        return QY

    
            
def find_B_rel(n, p, T):  
    """ Modeled for silicon only.
    Find radiative recombination decrease according to "Injection dependence of spontaneous radiative
    recombination in c-Si: experiment, theoretical analysis, and simulation by Altermatt" in 2005. 
    His paper has typos as noted in Altermatt's 2011 review page 325: replace b2 with 2b2, 
    b1 with 2b1, and n + p need brackets.
    
    Used by Flux and recombination.J_Auger_Richter. """

    r_max = .2
    s_max = 1.5e18
    w_max = 4e18
    s_min = 1e17
    w_min = 1e9
    b2 = .54
    r1 = 320
    s1 = 550
    w1 = 365
    b4 = 1.25
    r2 = 2.5
    s2 = 3
    w2 = 3.54  
    b_max = 1
    r_min = 0
    
    b_min = r_max + (r_min - r_max)/(1 + (T/r1)**r2)
    b1 = s_max + (s_min - s_max)/(1 + (T/s1)**s2)
    b3 = w_max + (w_min - w_max)/(1 + (T/w1)**w2)
    B_rel = b_min + (b_max - b_min)/(1 + ((n + p)/(2*b1))**(2*b2) + ((n + p)/b3)**b4)
    return B_rel




class Flux:
    def __init__(self, E1, E2, T, volt, photocollection, stack, note='', model=None):
        """ For net radiative recombination - radiation is emitted out the front and rear 

        Outputs flux in 1/(s m^2)"""
        
        EQE, rear_QE, nr = photocollection.EQE, photocollection.rear_QE, photocollection.nr   
        eVs_original = stack.eVs
        if model == 'absorptance' or stack.anything_variable[0]=='absorptance':  # if applying the absoprtance model for radiative recombination
            EQE = photocollection.absorptance
            rear_QE = photocollection.rear_emittance  
        
        # radiative recombinaation out the front
        [E, blackbody_photocurrent] = blackbody(volt, E1, E2, stack)   # E is in J whereas eVs is in J. Also, the array E is made with finer resolution than eVs, which is given as the AM1.5G resolution.
        EQE = np.interp(E, q*eVs_original, EQE)  # equal to absorptance if not diffusion or SRV limited
        front_flux = trapz(EQE*blackbody_photocurrent, E)  # photocurrent (s^-1 m^-2 eV^-1) 
        #
        # Special case: limited qcceptance qngles, ie angular selective filters
        if stack.acceptance_angle != np.pi/2:
            thetas = np.linspace(0, stack.acceptance_angle, 100)
            front_flux = 2*trapz(np.cos(thetas)*np.sin(thetas), thetas)*trapz(EQE*blackbody_photocurrent, E) # *2 because taking ratio of solid angle to hemisphere solid angle, and projected solid angle of hemisphere is 1/2
                                    # assumes absorptance/EQE is angle independent
                # Limited acceptance angles and maximum concentration possible X linked by 
                # X_max = Sin[theta]^2/Sin[4.65*^-3]^2 = 46250*Sin[theta]^2
        
        # radiative recombinaation out the front
        if np.any(stack.rear_reflectance<1):  # Emittance to backmirror        
            nr = np.interp(E, q*eVs_original, nr)
            rear_QE = np.interp(E, q*eVs_original, rear_QE)
            spectral_back_flux =  nr**2*rear_QE*blackbody_photocurrent
            back_flux = trapz(spectral_back_flux, E)     
        else: back_flux = 0 


        if stack.nonradiative_recombination_modeling == 'Yes':          
            if stack.composition[stack.layer_num] == 'Si':  # extra model availible for Si
                carriers = carrier_models.Carriers(volt, stack)
                B_rel = find_B_rel(carriers.n, carriers.p, T)
                front_flux = B_rel*front_flux
        flux = front_flux + back_flux
        self.flux = flux
        self.front_flux = front_flux
        self.back_flux = back_flux
        
  




class photon_recycling:
    def __init__(self, carriers, volt, photocollection, stack):
        """ Calculates probability of photon recycling.
        
        Used in recombination.py to calculate spectral diffusivity. 
        The EQE and absorptance models for Jrad implicity incorporate photon recycling.""" 
        
        
        if  stack.nonradiative_recombination_modeling=='Yes':
            T = stack.T_cell
            eVs = stack.eVs # Set of energies(eV)  
            nr = photocollection.nr 
            absorptance, rear_emittance, alpha, alpha_FCA, Rf_int, Rf_ext = (
                photocollection.absorptance, photocollection.rear_emittance, photocollection.alpha, photocollection.alpha_FCA, photocollection.Rf_int, photocollection.Rf_ext)
            E1_pr = max(eVs[np.where(alpha>0)[0][0]], volt+.005)
            E2_pr = 2*stack.bandgap + .5
            [E, blackbody_photocurrent] = blackbody(volt, E1_pr, E2_pr, stack)#n=5e3  # alpha has resolution at 2000

            spectral_radiative_coefficient = 4*(g*1e-4)/(carriers.ni_eff**2)*nr**2*(q*eVs)**2*alpha*np.exp(-q*eVs/(k*T))  # (cm^3/s/eV) 1e-4 to go um to cm   # 4 for all directions (sphere surface area = 4*circle)
            stack.spectral_B = spectral_radiative_coefficient
            radiative_coefficient = trapz(y=spectral_radiative_coefficient, x=q*eVs)  # B by the van Roosbroeck-Shockley equation. B Should be about 4.73e-15 cm^3/s for Si
            stack.B = radiative_coefficient

            alpha = np.interp(E, q*eVs, alpha) 
            alpha_FCA = np.interp(E, q*eVs, alpha_FCA) 
            nr = np.interp(E, q*eVs, nr) 
            Rf_int = np.interp(E, q*eVs, Rf_int)
            absorptance = np.interp(E, q*eVs, absorptance)  
            rear_emittance = np.interp(E, q*eVs, rear_emittance)  
            alpha_total = alpha + alpha_FCA + 1e-60
 
            internal_emission = 4*nr**2*alpha*blackbody_photocurrent            
            Pesc_front_E = 1/4*((1-Rf_int)/(1-Rf_ext))/(alpha_total*stack.thickness)*absorptance   # probability of escaping out the front   
            Pesc_rear_E = 1/4*1/(alpha_total*stack.thickness)*rear_emittance
            
            P_PR_E = (1 - (Pesc_front_E + Pesc_rear_E))/(1 + alpha_FCA/alpha)# probability of escaping           
            P_PR = trapz(P_PR_E*internal_emission, E)/trapz(internal_emission, E)           
            P_FCA_E =   P_PR_E*alpha_FCA/alpha  # this is the probability that internally emitted photons will be absrobed by free carrier absorption. Usually negligible.

            self.alpha = alpha
            self.alpha_total = alpha_total
            self.energies = E
            self.internal_emission = internal_emission          
            stack.P_PR_E = P_PR_E
            self.P_PR = P_PR
            self.P_FCA_E = P_FCA_E
            stack.P_PR = P_PR
            stack.photon_recycling = self
        else:
            self.P_PR = 'N/A'            