""" This module determines carrier concentration, diffusivity, and resistivity and is mostly used by recombination.py.

Created 07/06/2020 by Nicholas Irvin"""

import numpy as np
import math

# Constants
q = 1.60217646e-19  # Charge of electron (C)
k = 1.380658e-23  # J/K  Boltzmann's constant   
c = 2.99792458e8  # m/s  The speed of light
h = 6.6260755e-34  # J*s  Planck's constant (not h-bar)
g = 2*np.pi/(c**2)/(h**3)                
inf = float('inf')  # define infinity


def find_ni(T, stack): 
    """ intrinsic carrier concentration (cm^-3) """
    if stack.composition[stack.layer_num] == 'GaAs':
        ni = 1.79e6 # Shur, M., Physics of Semiconductor Devices?    some people use 2.07e6 b/c that fits their density of states value
    if stack.composition[stack.layer_num] == 'CdTe':
        ni = 8.19e5 # extrapolated from 1E6 at 300 K and "Temperature dependence of the band gap energy of crystalline CdTe"
    if stack.composition[stack.layer_num] == 'GaNP':
        ni = 288  # no ref    #2.7e6 # https://www.azom.com/article.aspx?ArticleID=8347
    if stack.composition[stack.layer_num] == 'CIS':
        ni = 8.4e9 # https://aip-scitation-org.ezproxy1.lib.asu.edu/doi/pdf/10.1063/1.4767120?class=pdf
    if stack.composition[stack.layer_num] == 'CIGS':
        ni = 1.2e9 # "Numerical Modeling of CIGS and CdTe Solar Cells: Setting the Baseline"
    if stack.composition[stack.layer_num] == 'perovskite triple cation':
        ni = 2.3e5  # from "Supplementary Material of Ref 12 of Benefit from Photon Recycling at the Maximum Power Point of State-of-the-Art Perovskite Solar Cells"
    if stack.composition[stack.layer_num] == 'perovskite MaPI':
        ni = 6e4  #"Optical and electrical optimization of all-perovskite pin type junction tandem solar cells" original source ref 33?
    if stack.composition[stack.layer_num] == 'Si':
        if T == 273.15 + 25:
            ni = 8.28e9 # Richter
        if T == 300:
            ni = 9.7e9  # 9.65 for PC1D?
        else:
            ni = 5.29e19*(T/300)**2.54*np.exp(-6726/T)
            # From Misiakos's "Accurate measurements of the silicon intrinsic carrier density from 78 to 340 K". 
            # Use Sproul instead from previous code version?:
    return ni


def fSchenk(Nd, Na, n, p, T):
    """ Schenk bandgap narrowing model. Bandgap narrows with injection."""
    #fundamental constants
    Ry = 1.655e-2
    a_ex = 3.7185e-7
    Tn = T*k/q/Ry
    #model parameters
    alpha_e = 0.5187 
    alpha_h = 0.4813 
    b_e = 8
    b_h = 1 
    c_e = 1.3346 
    c_h = 1.2365 
    d_e = 0.893 
    d_h = 1.153 
    g_e = 12 
    g_h = 4 
    h_e = 3.91 
    h_h = 4.2 
    j_e = 2.8585 
    j_h = 2.9307 
    k_e = 0.012 
    k_h = 0.19 
    p_e = 7/30
    p_h = 7/30 
    q_e = 0.75 
    q_h = 0.25
    #normalization to Bohr radius
    p_n = p*a_ex**3 
    n_n = n*a_ex**3 
    c = p_n + n_n 
    n_p = alpha_e*n_n + alpha_h*p_n 
    Nd_n = Nd*a_ex**3 
    Na_n = Na*a_ex**3 
    Nc = Nd_n + Na_n
    #exchange-correlation 
    xcn = ( (4*np.pi)**3*c**2*((48*n_n/(np.pi*g_e))**(1/3)+c_e*np.log(1+d_e*n_p**p_e)) + 8*np.pi*alpha_e/g_e*n_n*Tn**2 + np.sqrt(8*np.pi*c)*Tn**(5/2)  ) / ( (4*np.pi)**3*c**2 + Tn**3 + b_e*np.sqrt(c)*Tn**2 + 40*c**(3/2)*Tn   ) * Ry * 1000                                    
    xcp = ( (4*np.pi)**3*c**2*((48*p_n/(np.pi*g_h))**(1/3)+c_h*np.log(1+d_h*n_p**p_h)) + 8*np.pi*alpha_h/g_h*p_n*Tn**2 + np.sqrt(8*np.pi*c)*Tn**(5/2)  ) / ( (4*np.pi)**3*c**2 + Tn**3 + b_h*np.sqrt(c)*Tn**2 + 40*c**(3/2)*Tn   ) * Ry * 1000                                      
    #ionic
    iD = Nc *(1+c**2/Tn**3) / ( np.sqrt(Tn*c/(2*np.pi))*(1+ h_e*np.log(1+np.sqrt(c)/Tn)) + j_e*c**2/Tn**3*n_p**(3/4)*(1+k_e*n_p**q_e)  )  *Ry*1000
    iA = Nc *(1+c**2/Tn**3) / ( np.sqrt(Tn*c/(2*np.pi))*(1+ h_h*np.log(1+np.sqrt(c)/Tn)) + j_h*c**2/Tn**3*n_p**(3/4)*(1+k_h*n_p**q_h)  )  *Ry*1000

    BGN = (xcn + xcp + iD + iA)*1e-3
    return BGN




def find_ni_eff(volt, Nd, dn, T, stack):
    """ Consider bandgap narrowing to calculate ni_effective (Effective intrinsic carrier concentration) at 
    given injection levels"""
    
    # temporarily consider electron as majority carrier
    Na = 1
    ni = find_ni(T, stack)
    no = Nd/2 + np.sqrt((Nd/2)**2 + ni**2)
    po = ni**2/no
    dn = ( -no+ np.sqrt(no**2 + 4*(ni**2*np.exp(q*volt/(k*T)))) )/2
    n = no + dn
    p = po + dn
    # here we identify majority carrier
    if stack.dopant_type == 'n':
        pass
    if stack.dopant_type == 'p':
        [Na,Nd,p,n]  = [Nd,Na,n,p]  # switch the labels    
    
    # bandgap narrowing
    if stack.composition[stack.layer_num] == 'Si':        
        dEg = fSchenk(Nd,Na,n,p,T) # (eV) 
        ni_eff = ni*np.exp(dEg/(2*k*T/q))        
    elif stack.composition[stack.layer_num] == 'GaAs':  # https://aip-scitation-org.ezproxy1.lib.asu.edu/doi/pdf/10.1063/1.111110
                                                        # good graph of bandgap narrowing at https://pdf.sciencedirectassets.com/271530/1-s2.0-S0921452600X02190/1-s2.0-S092145260201428X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHEaCXVzLWVhc3QtMSJIMEYCIQCt1ZDZgZnpC2LomAiNL0%2FZA12SHKVeoTIe%2F1S48K5pBQIhAMwjBRiD5loZ2A5TFrV5VMzUfmJisrWa0K9IErCHLEegKrQDCBoQAxoMMDU5MDAzNTQ2ODY1IgyQ%2BHn06yiM3mWsRcwqkQPnnUWbQBEMyxi%2BDPAD2ZBCHd4lh6c5wWTty72qt2UQFUy4jicds4zaPEs5guugfwzQA9BSfxwkhFG1e3eKJVcRjg8ULIKkjHXrK1ntr5TN5TbwO%2FBAGJrhu0%2BBb2PB8XZB75cYtzFqdXZFcH1JEAUAXuPctbQ6rOk5DIRrWKq363TJqhYa5Kxk4IV5MlgjS3uAsxYYGGaUy8jTUg2zB%2FVL39%2F2tENAw836I1ynCrx8KIuTNVJ2jzp7FVZCSgovHdtskV%2FVOBhPi5lA6Yh8K4rmeHPg1N93czRNeqGaRbShBqRKV8e4JQU8cFnEmqgGVzafQ3kFDqW8RotU4Gzp%2FOA3%2F7WcTda%2BcxjlP6TkXzh5QmApE3PQtYQNI0fQrB6%2BoQuQ1pKE0i1mP5mdDgdmm4FnOm5EFpiJS08isOv6fcdX%2FF0te4tY7YPK10DrSfRCwo%2FvfW%2FHnZP0080n%2Br9XH5ufhYVgGwN0p1N5WBrYJlKrHiVZGGgYHlzmpy7fc82as2efS%2B1RZhd5JKJ3HquLqhagfDD89%2Bj%2BBTrqAWktt5JVi8eIeOO1R%2BsyoeyoxxyDB2mAuIhOsOTGAsjoBm%2Fk80HrGP0kQN3kUTKcibbIOJa9owarFgdxc3QxwpQMw9Pqg%2FfpE3u0Wnecel8iwjZo5mY3dkMMua4ynOcEGl4AcQUBP%2B0tMCRWPbmg4FaYccRykg1uQ5WxOLNjiqvlb%2BX4Yrrw9b0tvuYi1mRJgOLlT2LSa91ynaL9h8%2Bzg24p9kVoUgyP%2FQcvEQrxnmw8hrWqdjg0X6InWUnn%2BIchFONTJ%2BUiN18cRxAnB2%2BiL9KOzsOzLAU0pOnyQrsD%2Fogobc6uduR9%2BQCIBQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20201216T181129Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRLHUWI5Y%2F20201216%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=6d23f16aff91dcf990f40e8dcbce85d2d5a91b31890231096510eeb2976dbadf&hash=3019939120f624552380b7071f84e5ce43ad623135c737f7503de8782097860e&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S092145260201428X&tid=spdf-1f510e5f-b359-4fdb-8cf9-ce6bf5123591&sid=de8a18ee4acc4944019b81530e12e2e8299fgxrqa&type=client
        if stack.dopant_type == 'n':
            A = 3.23e-8
        if stack.dopant_type == 'p':
            A = 2.55e-8
        dEg = A*stack.dopant_density**(1/3)
        ni_eff = ni*np.exp(dEg/(2*k*T/q))        
    else: 
        ni_eff = ni
    return ni_eff


    

class Carriers:
    def __init__(self, volt, stack, ni_eff='get'):
        """ Compile information about carriers - all units of cm^-3. """
         
        ni = find_ni(stack.T_cell, stack)
        self.ni = ni
        # temporarily consider electron as majority carrier 
        Nd = stack.dopant_density 
        if ni_eff=='get':
            ni_eff = find_ni_eff(volt, Nd, 0, stack.T_cell, stack)
            self.ni_eff = ni_eff
        if ni_eff=='ni':
            ni_eff = ni
        self.Nd = Nd
        no = Nd/2 + np.sqrt((Nd/2)**2 + ni_eff**2)  # b/c Charge nuetrality
        po = ni_eff**2/no
        dn = ( - no + np.sqrt(no**2 + 4*(ni_eff**2*np.exp(q*volt/(k*stack.T_cell)))) )/2        # because nopo = ni^2, so np = ni^2e^(qV/kT)   = (no+dn)(0+dn)    # then use quadratic formula
        self.dn = dn
        if stack.dopant_type == 'n':  # save data
            self.n0 = no
            self.p0 = po       
            self.n = no + dn 
            self.p = po + dn
        if stack.dopant_type == 'p':  # switch labels and save data
            self.p0 = no
            self.n0 = po       
            self.p = no + dn 
            self.n = po + dn

        self.volt = volt
        
        

def mob_klaassen(stack, carriers):
    """Klaassen model
    Return the mobility (cm2/Vs)
    given the dopings and excess carrier concentration
    DOI: 10.1109/IEDM.1990.237157"""
    
    T = stack.T_cell # model designed for 298.16 K       
    if stack.dopant_type =='n':
        Nd = stack.dopant_density
        Na = 1e11
    if stack.dopant_type =='p':
        Na = stack.dopant_density
        Nd = 1e11
    p = carriers.p
    n = carriers.n     
    cc = p + n
        
    s1 = 0.89233
    s2 = 0.41372
    s3 = 0.19778
    s4 = 0.28227
    s5 = 0.005978
    s6 = 1.80618
    s7 = 0.72169
    r1 = 0.7643
    r2 = 2.2999
    r3 = 6.5502
    r4 = 2.367
    r5 = -0.01552
    r6 = 0.6478
    fCW = 2.459
    fBH = 3.828
    mh_me = 1.258
    me_m0 = 1
    cA = 0.5
    cD = 0.21
    Nref_A = 7.20E+20
    Nref_D = 4.00E+20
    Za_Na = 1 + 1 / (cA + (Nref_A / Na) ** 2)
    Zd_Nd = 1 + 1 / (cD + (Nref_D / Nd) ** 2)
    Na_h = Za_Na * Na
    Nd_h = Zd_Nd * Nd
    boron_µmax = 470.5
    boron_µmin = 44.9
    boron_Nref_1 = 2.23E+17
    boron_α = 0.719
    boron_θ = 2.247
    phosphorus_µmax = 1414
    phosphorus_µmin = 68.5
    phosphorus_Nref_1 = 9.20E16
    phosphorus_α = 0.711
    phosphorus_θ = 2.285

    µ_eN = phosphorus_µmax ** 2 / (phosphorus_µmax - phosphorus_µmin) * (T / 300) ** (3 * phosphorus_α - 1.5)
    µ_hN = boron_µmax ** 2 / (boron_µmax - boron_µmin) * (T / 300) ** (3 * boron_α - 1.5)
    µ_ec = phosphorus_µmax * phosphorus_µmin / (phosphorus_µmax - phosphorus_µmin) * (300 / T) ** 0.5
    µ_hc = boron_µmax * boron_µmin / (boron_µmax - boron_µmin) * (300 / T) ** 0.5
    Ne_sc = Na_h + Nd_h + p
    Nh_sc = Na_h + Nd_h + n
    PBHe = 1.36e+20 / cc * me_m0 * (T / 300) ** 2
    PBHh = 1.36e+20 / cc * mh_me * (T / 300) ** 2
    PCWe = 3.97e+13 * (1 / (Zd_Nd ** 3 * (Nd_h + Na_h + p)) * ((T / 300) ** 3)) ** (2 / 3)
    PCWh = 3.97e+13 * (1 / (Za_Na ** 3 * (Nd_h + Na_h + n)) * ((T / 300) ** 3)) ** (2 / 3)
    Pe = 1 / (fCW / PCWe + fBH / PBHe)
    Ph = 1 / (fCW / PCWh + fBH / PBHh)
    G_Pe = 1 - s1 / ((s2 + (1 / me_m0 * 300 / T) ** s4 * Pe) ** s3) + s5 / (((me_m0 * 300 / T) ** s7 * Pe) ** s6)
    G_Ph = 1 - s1 / ((s2 + (1 / (me_m0 * mh_me) * T / 300) ** s4 * Ph) ** s3) + s5 / (
            ((me_m0 * mh_me * 300 / T) ** s7 * Ph) ** s6)
    F_Pe = (r1 * Pe ** r6 + r2 + r3 / mh_me) / (Pe ** r6 + r4 + r5 / mh_me)
    F_Ph = (r1 * Ph ** r6 + r2 + r3 * mh_me) / (Ph ** r6 + r4 + r5 * mh_me)
    Ne_sc_eff = Nd_h + G_Pe * Na_h + p / F_Pe
    Nh_sc_eff = Na_h + G_Ph * Nd_h + n / F_Ph
    # Lattice Scattering
    µ_eL = phosphorus_µmax * (300 / T) ** phosphorus_θ
    µ_hL = boron_µmax * (300 / T) ** boron_θ
    µe_Dah = µ_eN * Ne_sc / Ne_sc_eff * (phosphorus_Nref_1 / Ne_sc) ** phosphorus_α + µ_ec * ((p + n) / Ne_sc_eff)
    µh_Dae = µ_hN * Nh_sc / Nh_sc_eff * (boron_Nref_1 / Nh_sc) ** boron_α + µ_hc * ((p + n) / Nh_sc_eff)

    µe = 1 / (1 / µ_eL + 1 / µe_Dah)
    µh = 1 / (1 / µ_hL + 1 / µh_Dae)
    return µe, µh




class Mobility:  # (cm^2 V^-1 s^-1)
    def __init__(self, carriers, stack):
        if stack.composition[stack.layer_num] == 'Si':
            mobility_e, mobility_h =   mob_klaassen(stack,carriers=carriers)
        if stack.composition[stack.layer_num] == 'GaAs':
            # from Empirical low-field mobility model for III-V compounds applicable in device simulation codes
            mu_max = 9400  # electron mobility
            mu_min = 500
            Nr = 6e16
            a = 0.394
            mobility_e = mu_min + (mu_max*(stack.T_cell/300)**2.1 - mu_min)/(1+(stack.dopant_density/(Nr*(stack.T_cell/300)**3))**a)
            mu_max = 491.5  # hole mobility
            mu_min = 20
            Nr = 1.48e17
            a = 0.394   
            mobility_h = mu_min + (mu_max*(stack.T_cell/300)**2.1 - mu_min)/(1+(stack.dopant_density/(Nr*(stack.T_cell/300)**3))**a)
        if stack.composition[stack.layer_num] == 'CIGS' or (stack.composition == 'CIS'):   # These are thin film values, single crystal values are higher
            mobility_e = 200  # https://aip-scitation-org.ezproxy1.lib.asu.edu/doi/pdf/10.1063/1.366365 
            mobility_h = 30  # https://books.google.com/books?id=YhCQDwAAQBAJ&pg=PA21&lpg=PA21&dq=cigs+hole+mobility&source=bl&ots=Va2kX3Op-s&sig=ACfU3U0yc_QXlNpbx5bzTsMzF533890u6Q&hl=en&sa=X&ved=2ahUKEwjd88PS4ejpAhVKIDQIHWwbBmc4ChDoATACegQIChAB#v=onepage&q=cigs%20hole%20mobility&f=false
    
        if stack.composition[stack.layer_num] == ('CdTe'):   # These are thin film values, single crystal values are higher
            mobility_e = 10  # https://pdf.sciencedirectassets.com/280687/1-s2.0-S2211379719X00029/1-s2.0-S2211379719308757/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjENL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIG6pKBI6zF%2BEy27veoIVIk5qRoimnXVB%2B2q4%2FRruch%2B%2BAiEA3HYiTOJBaL54wYr0ydX%2BM6k4MScKHu2haxs0AZSE1PUqtAMISxADGgwwNTkwMDM1NDY4NjUiDAZDoPQifSuuCwHg3CqRA8ZpncdXkoc%2BNn6YIfCTrB9p8Q%2FL%2BQga17ePsmdfwuFJa4I%2FJ%2BM%2FfsXDXSHAEMUTZoNLajZN8gH%2FYBQvlZfxpKwGtC3siyvR7Fk25jb2MWpidXhqHLLt%2BVKAlmrMkQlh1JpafRclCqFWbLAN3m97VzHkBPA6%2FAaDYMeHWBRNZuuQvN41doBbmMCXsE1J0r%2F1%2FT8UrvuBG7MGtjzJPqoaYwpqIuD4z8kRURh4DLVnez1o19U3IIqRbFmukoRPdGMGkfRf3anEySFjQgS69XflHyrXVmdc%2FAuP01X4c7ahuteadCm4Z5s2ZpO6WyiF5mMpjyQnyHAgseFf1oRG6129shAFYKgZNehsbKcyCNyeLYkFCLVyM%2BNusg%2F7%2FjLMZH7zweI0DS%2FLT5YNd8NkyEmBWv4UCASCzKpk9AxB6bX5jdO78yjsWalEuqOZF%2Bii6uuiuNmrgP1DyiPE4vnQX9d2uf6RNVqy7hrHr54OJytXXXA%2Fe2yMf2%2BObbU%2FsqdMInlnR7fjh91d%2BZeb%2Fe3eVzBtZfejMPXhifcFOusBCrIEeaLMrQv6n1NOL4tTJt50dauNeRnfnPrXzRD74%2BsUe6PZVA7TTkEEroiMjyaDALgI4sKIKryr5Dgy%2FQbLd0o%2Fbi2tZAwI8tV5nM54y3oMvHntRr6A%2BTWYk12UkzgKEq7qRDDoJF3oqQzUcbg1sRtljLTrd%2F02wcDjD%2BrDsTtF55yydCTTIZjg4jeB1EbqKbyDlNA6T1y4K5ImwX1nmZ326HqMmu3q7hMHa74SqIXpTkyscQwpWNGgN1qlULrbNR8KchX3k4pKJUfFQSfhjkBEGK8BBD%2FUv7bEVeCHIeJ3eHC%2BxUnshWO6yA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200611T190219Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3VQ7VWMU%2F20200611%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ff740b40d2aa7555aa1ea52d90e1b649d75d83154d013015068aac26414d1dd7&hash=7d65e74f14ac6c7951693eb93e8b979ac6af13a3879425d8ddf2e82774ea9b97&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2211379719308757&tid=spdf-93fff349-7805-4238-bf37-b83c9abc2e2e&sid=0ee4cbcc88c20542d858f9c76319dd518ef9gxrqa&type=client
            mobility_h = 1  # https://aip-scitation-org.ezproxy1.lib.asu.edu/doi/pdf/10.1063/1.4891846
            # Glockler has mu_e = 320, mu_h = 40 in "Setting the Baseline"
            # perpindicular mobility is what matters and will be higher https://www.osti.gov/pages/servlets/purl/1371645
            # crystalline mobilities are much higher  https://ieeexplore-ieee-org.ezproxy1.lib.asu.edu/stamp/stamp.jsp?tp=&arnumber=1305328   

        if stack.composition[stack.layer_num] == ('perovskite triple cation'):   
            mobility_e = 1  # from(thin film)  - from averging e mobilities in MaPI and FaPbI actually from Charge-Carrier Mobilities in Metal Halide Perovskites: Fundamental Mechanisms and Limits
            mobility_h = 2  # - from averging e mobilities in MaPI and FaPbI 
            # but mobility in perovskites is anisotropic 
            # range of MAPI and FaPbI e or h is 0.2, 33

        if stack.composition[stack.layer_num] == ('perovskite MaPI'):   
            mobility_e = 1  # from https://pubs.acs.org/doi/pdf/10.1021/acsenergylett.7b00276
            mobility_h = 0.7

        if stack.composition[stack.layer_num] == ('GaNP'):   # These are thin film values, single crystal values are higher  
            mobility = 'bad'
            if stack.dopant_type == 'p':
                Nh = stack.dopant_density
                Ne = 1e12
            if stack.dopant_type == 'n':
                Ne = stack.dopant_density
                Nh = 1e12
            if mobility == 'good':  # from User inputs at top of file
                # did h and e got mixed up??
                mobility_h = 30 - 20/4*(math.log(Nh, 10) - math.log(10**15, 10)) # 'n': 55, 'p': 10 # 30 - 20/4*(math.log(Ne, 10) - math.log(10**15, 10)) # 'p'
                mobility_e = 180 - 130/3*(math.log(Ne, 10) - math.log(10**16, 10)) # 'n': 180, 'p': 30 # 180 - 130/3*(math.log(Nb, 10) - math.log(10**16, 10)) # 'n'
            if mobility == 'bad':
                mobility_h = 10 - 5/4*(math.log(Nh, 10) - math.log(10**16, 10))
                mobility_e = 60 - 50/3.5*(math.log(Ne, 10) - math.log(10**15, 10))
        self.mobility_e = mobility_e
        self.mobility_h = mobility_h



def find_diffusivity(carriers, stack): # returns D from Einstein's relation (in cm^2/s)
    if stack.dopant_type == 'p':
        mobility = Mobility(carriers, stack).mobility_e
    if stack.dopant_type == 'n':
        mobility = Mobility(carriers, stack).mobility_h
    # mobility = stack.anything_variable  # can change the mobility here
    diffusivity = k*stack.T_cell/q*mobility
    return diffusivity
        
            
def base_resistivity(carriers, stack):
    """ Calculate base resistance as a function of carrier concentration in the base. 
    This calculation needed to incorporate the benefits of doping the base. 
    Outputs resistance in ohms*cm."""
    
    mob = Mobility(carriers, stack)
    return (q*(carriers.n*mob.mobility_e + carriers.p*mob.mobility_h))**-1  # ohms*cm


