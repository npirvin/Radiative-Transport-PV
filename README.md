# Radiative-Transport-PV

This program was used to calculate results for the 2022 paper "Feedback between radiation and transport in photovoltaics." 
This work merges detailed-balance calculations with the      drift-diffusion model. 

Most parameters can be specified in the file run.py, which is also where calculations are run.
Here is the order that files are called:
run -> input_output_management -> stack_power -> single_cell_power -> recombination -> spectral -> carrier_models/sunlight.         
Many parameters can be turned off but still need a placeholder. For example, if file says MEG = 'No', 
    then max_yield and threshold_energy_normalized will not matter (but they still need to be defined in order to run the next file.)

This was written on python 3.10. The code also uses standard python libraries, such as numpy. 
These libraries are availible by installing the python distribution Anaconda.

Note: This code originally targeted general detailed balance efficiencies before being changed to focusing on 
    feedback between transport and radiation. As such, the multijunction features and the multi-exciton generation (MEG) 
    may not be compatible with the new drift-diffsion model. Using the calculations for multijunctions or MEG 
    may conflict with more recent parts of the code, and thus need reverification before use.


Nicholas P. Irvin, 1/10/2022
