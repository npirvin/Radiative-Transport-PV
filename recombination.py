""" This module determines Auger, radiative, surface, and trap-assited recombination.
Used primarily by find_current function in single_cell_power.
Uses spectral.py to get radiative recombination and carriers.py for carrier concentration.

Created 12/18/2019 by Nicholas Irvin"""

import numpy as np
import math
from scipy.integrate import trapz
import spectral, carrier_models


# Constants
q = 1.60217646e-19  # Charge of electron (C)
k = 1.380658e-23  # J/K  Boltzmann's constant   
c = 2.99792458e8  # m/s  The speed of light
h = 6.6260755e-34  # J*s  Planck's constant (not h-bar)
g = 2*np.pi/(c**2)/(h**3)                
inf = float('inf')  # define infinity




def Auger(volt, photocollection, stack, P_PR):
    """ Calculate radiative + Auger-Meitner recombination currents for Richter parametrization 
    at room temperature. Voltage in volts, thickness in cm, cell temperature in kelvin.
    Outputs recombination in cm^3 s^-1 at 300 K."""
    
    if (298 < stack.T_cell < 300.1) and stack.composition[stack.layer_num] == 'Si':  
        return Auger_Richter(volt, photocollection, stack, P_PR)

    if stack.composition[stack.layer_num] == 'Si':
        if 273.15 <= stack.T_cell <= 300:
            Ca = 1.66e-30 # (cm^6/s) Amipolar Auger coefficient according to Sinton (Recombination ir Highly Injected Silicon), but 
            # value overestimates lifetimes for lowly doped silicon according to Richter
        else:
            Ca = 1.1e-28/(stack.T_cell-193) + 2.1e-33*stack.T_cell # (cm^6/s) Amipolar Auger coefficient 
            # To calculate Auger recombination as function of temeprature from Sisi Wang 2012 parametrization. This parametrization was only done for 243 to 473 K with injection level of 5*10^16 cm^-3"""
    if stack.composition[stack.layer_num] == 'GaNP':
        Ca = 1e-30 # is value for GaP http://www.ioffe.ru/SVA/NSM/Semicond/GaP/electric.html
    if stack.composition[stack.layer_num] == 'CdTe':
        Ca = 9e-32  # (cm^6/s) https://www-osapublishing-org.ezproxy1.lib.asu.edu/DirectPDFAccess/1E8B0161-98FF-B1DC-5B7839CB2DAD90A3_309963/oe-23-2-1256.pdf?da=1&id=309963&seq=0&mobile=no Time resolved photo-luminescent decay characterization of mercury cadmium telluride focal plane arrays or 6e-32 https://www.osti.gov/pages/servlets/purl/1419412    
    if stack.composition[stack.layer_num] == 'CIS':
        Ca = 6e-30  # (cm^6/s) https://onlinelibrary-wiley-com.ezproxy1.lib.asu.edu/doi/epdf/10.1002/pssc.200778414 Is Auger recombination responsible for the efficiency rollover
    if stack.composition[stack.layer_num] == 'CIGS':
        Ca = 1.2e-30  # (cm^6/s) https://onlinelibrary-wiley-com.ezproxy1.lib.asu.edu/doi/epdf/10.1002/pssc.200778414 Is Auger recombination responsible for the efficiency rollover
    if stack.composition[stack.layer_num] == 'perovskite triple cation':
        Ca = 1e-28  # (cm^6/s)  from Supplementary Material of "Benefit from Photon Recycling at the Maximum Power Point of State-of-the-Art Perovskite Solar Cells"
    if stack.composition[stack.layer_num] == 'perovskite MaPI':
        Ca = 5.4e-28  # (cm^6/s)  from "Hybrid Perovskite Films Approaching the Radiative Limit With Over 90% Photoluminescence Quantum Efficiency."        
    if stack.composition[stack.layer_num] == 'GaAs':
        Ca = 7e-30  # (cm^6/s) Amipolar Auger coefficient from Strauss's 1993 “Auger recombination in intrinsic GaAs”
    W = stack.thickness
    carriers = carrier_models.Carriers(volt, stack)
    if volt >.01:
        if stack.dn_z[0] != 1:
            dn_z = stack.dn_z
            Z = stack.Z
            n = carriers.n - carriers.dn + dn_z
            p = carriers.p - carriers.dn + dn_z

            J_Auger = q*Ca*trapz((n**2*p - carriers.n0**2*carriers.p0
                    + n*p**2 - carriers.n0*carriers.p0**2), Z)/2*1e4  # 1e4 for units of A/m^2.  with undoped: n^2p ~ ni^3*e^(3qV/2kT)
            tau_Auger = trapz(dn_z, Z)/(J_Auger*1e-4/q)  # 1e4 for units of A/m^2.  with undoped: n^2p ~ ni^3*e^(3qV/2kT)
        else:
            J_Auger = q*Ca*W*(carriers.n**2*carriers.p - carriers.n0**2*carriers.p0
                    + carriers.n*carriers.p**2 - carriers.n0*carriers.p0**2)/2*1e4  # 1e4 for units of A/m^2.  with undoped: J_Auger = q*Ca*W*carriers.ni**3*np.exp(3*q*volt/(2*k*stack.T_cell))*1e4
            tau_Auger = q*W*carriers.dn/(J_Auger*1e-4)  
    else:
        J_Auger = 0
        tau_Auger = inf
            
    return J_Auger, tau_Auger # (A/m^2)





# special Auger function only for Si
def Auger_Richter(volt, photocollection, stack, P_PR):
    """ Calculate radiative + Auger recombination currents for Richter parametrization 
    at room temperature. Voltage in volts, thickness in cm, cell temperature in kelvin.
    Outputs recombination in cm^3 s^-1 at 300 K. 
    
    From 'Improved quantitative description of Auger recombination in crystalline silicon.'"""

    carriers = carrier_models.Carriers(volt, stack)
    ni_eff = carrier_models.find_ni_eff(volt, carriers.Nd, 0, stack.T_cell, stack)
     
    N0_eeh = 3.3e17 
    N0_ehh = 7e17 
    g_eeh = 1 + 13*(1-np.tanh((carriers.n0/N0_eeh)**0.66 )) 
    g_ehh = 1 + 7.5*(1-np.tanh((carriers.p0/N0_ehh)**0.63))
   
    # # If trying to match doing the Richter's parametrization (instead of calculating radiative recombination), uncomment this block
    # B_low = 4.73e-15  # (cm^3/s) Low carrier injection and doping value for radiative
    # U = (U_Auger + (1-P_PR)*(carriers.n*carriers.p - ni_eff**2)*B_low*B_rel(carriers.n, carriers.p, stack.T_cell)
    #     + carriers.dn/stack.trap_lifetime)
    ## and also zero out radiative recombination in  the line "J_radiative = q*flux.flux #  A/m^2"
    if volt >.01:
        if stack.dn_z[0] != 1:
            dn_z = stack.dn_z
            Z = stack.Z
            n = carriers.n - carriers.dn + dn_z
            p = carriers.p - carriers.dn + dn_z

            U_Auger = (n*p - ni_eff**2)*(2.5e-31*g_eeh*carriers.n0 
                + 8.5e-32*g_ehh*carriers.p0 + 3e-29*dn_z**0.92) 
            J_Auger = q*trapz(U_Auger, Z)*1e4   # (A/m^2) 
            tau_Auger = trapz(dn_z, Z)/(J_Auger*1e-4/q)  # 1e4 for units of A/m^2.  with undoped: n^2p ~ ni^3*e^(3qV/2kT)
        else:
            U_Auger = (carriers.n*carriers.p - ni_eff**2)*(2.5e-31*g_eeh*carriers.n0 
                + 8.5e-32*g_ehh*carriers.p0 + 3e-29*carriers.dn**0.92) 
            J_Auger = q*stack.thickness*(U_Auger*1e4)   # (A/m^2) 
            tau_Auger = carriers.dn/(J_Auger*1e-4/(q*stack.thickness))  # 1e4 for units of A/m^2.  with undoped: n^2p ~ ni^3*e^(3qV/2kT)
    else:
        J_Auger = 0
        tau_Auger = inf
    return J_Auger, tau_Auger
   



def trap_assisted_recombination(volt, stack):
    if volt >.01:
        carriers = carrier_models.Carriers(volt, stack)
        if stack.dn_z[0] != 1:
            dn_z = stack.dn_z
            Z = stack.Z
            n = carriers.n - carriers.dn + dn_z
            p = carriers.p - carriers.dn + dn_z
            J_trap = (q*
                trapz((p*n - carriers.ni_eff**2)/(stack.trap_lifetime*(n + carriers.ni_eff + p + carriers.ni_eff)),Z)*1e4) # (A/m^2) trap-assisted recombination current                        
             
        else:
            n = carriers.n
            p = carriers.p
            J_trap = (q*stack.thickness*
                (p*n - carriers.ni_eff**2)/(stack.trap_lifetime*(n + carriers.ni_eff + p + carriers.ni_eff))*1e4) # (A/m^2) trap-assisted recombination current                                    
    else:
        J_trap = 0
    return J_trap
            



class Recombination:
    def __init__(self, volt, E1, E2, photocollection, stack):
        """ Compile different types of recombination."""
        
        W = stack.thickness
        stack.dn_z = np.ones(101)
        if stack.nonradiative_recombination_modeling == 'Yes' and volt!=0:
            carriers = carrier_models.Carriers(volt, stack)
            photon_recycling = spectral.photon_recycling(carriers, volt, photocollection, stack)
            alpha, alpha_total, PR, P_PR_E, E, internal_emission = photon_recycling.alpha, photon_recycling.alpha_total, stack.P_PR, stack.P_PR_E, photon_recycling.energies, photon_recycling.internal_emission  # average distance traveled of recycled photons
            
        def radiative(self, stack):  # store net radiative recombinations emitted out front and rear
            flux = spectral.Flux(E1, E2, stack.T_cell, volt, photocollection, stack)
            J_radiative = q*flux.flux #  A/m^2
            self.J_radiative = J_radiative  # A/m^2
            J_rad_front = q*flux.front_flux   # A/m^2
            self.J_rad_front = J_rad_front  # A/m^2
            self.J_rad_back = q*flux.back_flux  # A/m^2
             
            if stack.nonradiative_recombination_modeling == 'Yes' and volt!=0:
                PR = stack.photon_recycling
                self.J_FCA = q*W*trapz(PR.internal_emission*PR.P_FCA_E, PR.energies)   
                # need to interpolate arrays fromstack. eVs to E
                # self.J_FCA = q*W*trapz((photocollection.EQE/photocollection.absorptance)*PR.internal_emission*PR.P_FCA_E, PR.energies)
                self.J_radiative += self.J_FCA
    
        radiative(self, stack) 
        J_rec = self.J_radiative
        def find_rad_lifetime():  # (s)            
            if volt < .01:
                self.radiative_lifetime = inf
                return inf
            else:
                if stack.dn_z[0]!=1 and stack.anything_variable[0] != 'absorptance':  # first iteration or if using the absorptance model
                    dn = trapz(stack.dn_z, stack.Z)/W
                else:
                    dn = carrier_models.Carriers(volt, stack).dn 
                tau_rad = inf  if(dn==0) else(q*W*dn/(self.J_radiative)*1e4)
                self.radiative_lifetime = tau_rad
                return tau_rad
            
        if stack.nonradiative_recombination_modeling == 'No' or volt==0:
            self.carriers = 'N/A'
            if stack.nonradiative_recombination_modeling == 'Yes': 
                self.carriers = carrier_models.Carriers(volt, stack) 
            self.dn = 0
            JAuger = 0
            self.J_Auger = 0
            self.J_FCA = 0
            self.J_trap = 0
            self.J_SRV = 0
            SRV_lifetime = inf
            trap_lifetime = inf
            self.Auger_lifetime = inf
            self.radiative_lifetime = inf
            radiative_lifetime = inf
            self.diffusion_length = 1e9
            self.rad_diffusion_length = 1e9
            self.diffusivity = 1e9
            self.P_PR = 0
            self.base_resistivity = 0
        else:
            self.base_resistivity = carrier_models.base_resistivity(carriers, stack)
            self.carriers = carriers
            self.dn = carriers.dn
            self.P_PR = photon_recycling.P_PR
            radiative_lifetime = find_rad_lifetime()

            # now, photon-recycling diffusivity
            if stack.diffusion_limited == 'Yes':
                electrical_diffusivity = carrier_models.find_diffusivity(carriers, stack)
                def diffusivity_prefactor(z_limit):
                    # outputs the diffusivity prefactor as a function of the z limit. This is 1/3 in Dumke (1957), but here we limit the limits of the integral from infinity to z limit, which is related to absorber thickness.
                    # for speed, this was fit to the integral of (z*np.exp(-L)*(L-z)/L), with z from 0 to z_limit, and L from z to infinity
                    C = 0.357
                    Q = 1.75
                    B = 2.735
                    v = 1.968
                    def fitted_function(z):
                        return 1/3*C**(1/v)/(C+Q*z**(-B))**(1/v) # fits well above z_limit = .15
                    return np.piecewise(z_limit, [z_limit< 0.15, z_limit>= 0.15], [lambda z_limit: 0.5*(z_limit)**2, fitted_function])  # 4.99994*1e-1*(z_limit)**2  fits well below z_limit = 0.35

                diffusivity_Dumke_factor = diffusivity_prefactor(alpha_total*W/2)  
                spontaneous_radiative_lifetime = radiative_lifetime*(1-PR)
                spectral_diffusivity_Dumke = P_PR_E/spontaneous_radiative_lifetime*diffusivity_Dumke_factor*1/alpha_total**2   *(alpha/alpha_total)  # (cm^2/s)
                photon_recycling_diffusivity = trapz(spectral_diffusivity_Dumke*internal_emission, E) / trapz(internal_emission, E)  
                self.photon_recycling_D = photon_recycling_diffusivity   
                self.diffusivity = electrical_diffusivity + photon_recycling_diffusivity

            """ Effective lifetime"""
            if stack.lifetimes != []:  # in run.py, option to specific lifetimes, bulk_lifetimes, or trap_lifetimes
                J_rec += q*stack.thickness*carriers.dn/stack.lifetime*1e4                # 1e4 converts 1/cm^2 to 1/m^2 
                self.J_Auger = 0
                self.J_trap = J_rec - self.J_radiative
                self.J_SRV = 0
                trap_lifetime = 1/(1/stack.lifetime - 1/radiative_lifetime)
                self.Auger_lifetime = inf
                SRV_lifetime = inf
                
            else:
                """ SRV """
                if stack.SRVs != [] and stack.SRVs != [0]:
                    # From Dimensionless solution of the equation describing the effect of surface recombination on carrier decay in semiconductors:
                    SRV_lifetime = W/(stack.SRV) + (2*W/np.pi)**2/self.diffusivity 
                    # Assumes S2 = 0, ie one relatvely perfect contact
                    J_SRV = q*stack.thickness*carriers.dn/SRV_lifetime*1e4  # Surface
                    self.J_SRV = J_SRV  # A/m^2
                    J_rec += J_SRV
                else:
                    SRV_lifetime = inf
                    self.J_SRV = 0
                    
                """ Bulk """
                if stack.bulk_lifetimes != []:
                    bulk_lifetime = stack.bulk_lifetimes[stack.layer_num]
                    J_bulk = q*stack.thickness*carriers.dn/stack.bulk_lifetime*1e4
                    J_rec += J_bulk
                    self.J_Auger = 0
                    self.J_trap = 0
                    trap_lifetime = 1/(1/bulk_lifetime - 1/radiative_lifetime - 1/SRV_lifetime)
                    self.Auger_lifetime = inf
                    SRV_lifetime = inf
                    
                    """ Auger """
                else: 
                    auger = Auger(volt, photocollection, stack, P_PR=self.P_PR)
                    JAuger = auger[0]
                    self.Auger_lifetime = auger[1]
                    self.J_Auger = JAuger  # A/m^2   
                    J_rec += JAuger
                    
                    """ trap """   
                    trap_lifetimes= stack.trap_lifetimes
                    if trap_lifetimes == []:
                        trap_lifetime = inf
                        self.J_trap = 0
                    else:
                        trap_lifetime = stack.trap_lifetime  # used as fixed parameter
                        # from https://ieeexplore-ieee-org.ezproxy1.lib.asu.edu/stamp/stamp.jsp?tp=&arnumber=97400&tag=1
                        J_trap = (q*stack.thickness*
                                  (carriers.p*carriers.n - carriers.ni_eff**2)/(trap_lifetime*(carriers.n + carriers.ni_eff + carriers.p + carriers.ni_eff))*1e4) # (A/m^2) trap-assisted recombination current                        
                        J_rec += J_trap
                        self.J_trap = J_trap  # A/m^2
        
        def find_effective_lifetime(self, stack):
            if (stack.nonradiative_recombination_modeling=='No') or [trap_lifetime, self.radiative_lifetime, self.Auger_lifetime] == [inf, inf, inf]:
                self.bulk_lifetime = 1e9
                self.lifetime = 1e9 # effective lifetime
            else: 
  
                if volt == 0:  # at V=0,  the bulk lifetime becomes calculable, so take an arbitrary value
                    self.bulk_lifetime  = 1e9
                else:
                    self.bulk_lifetime = 1/(1/self.radiative_lifetime + 1/self.Auger_lifetime + 1/trap_lifetime)             # 1e-4 converts 1/cm^2 to 1/m^2 
                self.lifetime = 1/(1/trap_lifetime + 1/self.radiative_lifetime + 1/self.Auger_lifetime + 1/SRV_lifetime)
            if(stack.nonradiative_recombination_modeling == 'Yes') and (stack.diffusion_limited=='Yes') and self.diffusivity<1e8:
                self.diffusion_length = np.sqrt(self.bulk_lifetime*self.diffusivity)  # (cm))
                self.electrical_diffusion_length = np.sqrt(self.bulk_lifetime*electrical_diffusivity)  # (cm))
                self.ideal_electrical_diffusion_length = np.sqrt(radiative_lifetime*self.diffusivity)  # (cm))   is the radiative lifetime negative still?
                self.photon_recycling_L = np.sqrt(photon_recycling_diffusivity*self.bulk_lifetime)                
                 
                if stack.diffusion_limited == 'Yes':
                    stack.put_voltage_dependent_Jgen('On')    # if remotely diffusion limit, then recalcualte parameters with voltage
                elif stack.dopant_density>1e15:  # if can calculate free-carrier absorption FCA, then recalcualte parameters with voltage 
                    stack.put_voltage_dependent_Jgen('On')  # just stack.voltage_dependent_Jgen = 'On'   
                else:
                    stack.put_voltage_dependent_Jgen('Off')
            else:  # ignore difussion
                self.diffusion_length = 1e9  # (cm))
                self.electrical_diffusion_length = 1e9  # (cm))                      
                self.rad_diffusion_length = 1e9  # (cm)) 
                self.diffusivity = 1e9
                self.ideal_electrical_diffusion_length = 1e9
                self.photon_recycling_D = 1e9
                self.photon_recycling_L = 1e9

        find_effective_lifetime(self, stack)
        # loop between diffusion length and radiative lifetime to satisfy interdependance
        if stack.diffusion_limited == 'Yes':
            for i in range(100):
                old_radiative_lifetime = radiative_lifetime
                photocollection = spectral.Photocollection(E1, stack, volt=volt, diffusion_length=self.diffusion_length, diffusivity=self.diffusivity, absorptance=photocollection.absorptance, rear_emittance=photocollection.rear_emittance, rec=self)    
                radiative(self, stack)  # need to re add in J_rad
                radiative_lifetime = find_rad_lifetime()
                                
                # redo Auger
                auger = Auger(volt, photocollection, stack, P_PR=self.P_PR)
                self.Auger_lifetime = auger[1]
                self.J_Auger = auger[0]  # A/m^2   
 
                # redo trap-assited recombination (SRH)
                self.J_trap = trap_assisted_recombination(volt, stack)
                
                find_effective_lifetime(self, stack)  # recalculate L
                # On recalculation, Si and GaAs typically converges within one loop, CdTe takes 15 loops sometimes.
                if math.isclose(old_radiative_lifetime, radiative_lifetime, rel_tol=1e-4):
                    break
        # J_rec = self.J_radiative + J_nonrad
        J_rec = self.J_radiative  + self.J_Auger + self.J_trap + self.J_SRV
        if stack.nonradiative_recombination_modeling == 'No':
            J_rec = self.J_radiative/stack.fc_rec_ratio  # simple nonradiative recombination factor fc_rec_ratio is the ratio of net radiative recombination to net recombination
        
        self.J_recombination = J_rec 
        self.ERE = self.J_rad_front/J_rec # External Radiative Efficiency
        self.trap_lifetime = trap_lifetime
        self.SRV_lifetime = SRV_lifetime
        self.radiative_lifetime = radiative_lifetime
        stack.rec = self 
        stack.photocollection = photocollection        

