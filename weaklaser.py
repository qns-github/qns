# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import math
import numpy as np
import simpy
import itertools as it
from iteration_utilities import deepflatten
from photon import *
from photon_enc import *
from quantum_state import *
from component import *
from laser import *
from variable_ND_filter import *

class Weaklaser(Laser,NDFilter):
    
    """
    Models a Weak Laser [Laser + ND Filter(s)]
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator 
        freq (float) = Frequency with which the photons are emitted by the Laser
        wl (float) = Wavelength of the Laser
        lwidth (float) = Linewidth of the Laser
        twidth (float) = Temporal Width of the Laser
        mu_photons (int) = Mean Number of Photons emitted by the Laser
        enc_type (str) = Type of Quantum Information Encoding (see 'photon_enc.py') of the Photons emitted by the Laser
        noise_level (float) = Probability of the Quantum State of the Photons emitted by the Laser being altered because of Noise
        gamma (float) = probability of losing a photon
        lmda (float) =  probability of a photon getting scattered from the system (without any loss of energy)
        theta_FWHM (float) = Divergence Angle at Full Width at Half Maximum (FWHM) of the Laser
        height (float) = Height of the Laser
        length (float) = Length of the Laser
        min_OD (float) = Minimum Value of the Optical Density (OD) that the ND Filter can be set to
        max_OD (float) = Maximum Value of the Optical Density (OD) that the ND Filter can be set to
        thickness (float) = Thickness of the ND Filter
        max_num_of_ND_filters (int) = Maximum Number of ND Filters which can be used for attentuation of the Laser's output
        ND_filter_stack (list[float]) = OD(s) of the ND Filter(s) to be used for attentuating the Laser's output to single photon levels
        height_tot (float) = Total Height of the Weak Laser [Height of the Laser]
        length_tot (float) = Total Length of the Weak Laser [Length of the Laser + Thickness of the ND Filter(s) used]
    """

    def __init__(self,uID,env,freq,wl,lwidth,twidth,mu_photons,enc_type,noise_level,gamma,lmda,theta_FWHM,height,length,min_OD,max_OD,thickness,max_num_of_ND_filters):
        
        """
        Constructor for the Weaklaser class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
            freq (float) = Frequency with which the photons are emitted by the Laser
            wl (float) = Wavelength of the Laser
            lwidth (float) = Linewidth of the Laser
            twidth (float) = Temporal Width of the Laser
            mu_photons (int) = Mean Number of Photons emitted by the Laser
            enc_type (str) = Type of Quantum Information Encoding (see 'photon_enc.py') of the Photons emitted by the Laser
            noise_level (float) = Probability of the Quantum State of the Photons emitted by the Laser being altered because of Noise
            gamma (float) = probability of losing a photon
            lmda (float) =  probability of a photon getting scattered from the system (without any loss of energy)
            theta_FWHM (float) = Divergence Angle at Full Width at Half Maximum (FWHM) of the Laser
            height (float) = Height of the Laser
            length (float) = Length of the Laser
            min_OD (float) = Minimum Value of the Optical Density (OD) that the ND Filter can be set to
            max_OD (float) = Maximum Value of the Optical Density (OD) that the ND Filter can be set to
            thickness (float) = Thickness of the ND Filter
            max_num_of_ND_filters (int) = Maximum Number of ND Filters which can be used for attentuation of the Laser's output
        """
        
        Laser.__init__(self,uID,env,freq,wl,lwidth,twidth,mu_photons,enc_type,noise_level,gamma,lmda,theta_FWHM,height,length)
        NDFilter.__init__(self,uID,env,min_OD,max_OD,thickness)
        self.uID = uID
        self.max_num_of_ND_filters = max_num_of_ND_filters
        self.height_tot = height
        self.length_tot = length 
        self.ND_Filter_Stack = []

    def emit_and_attenuate(self,qs_list,basis):
        
        """
        Instance method which attentuates the output of the laser to ~ single photon levels using ND Filter(s)
        
        Details:
            The OD to which the ND Filter is to be set (so as to attentuate the laser's output to single photon levels) is decided on the basis of the mean number of photons emitted by the laser
            If this required OD value > maximum OD of a single ND Filter, then a stack of ND Filters are used to achieve the required OD value upto a pre-specified level of tolerance 
            The stack of ND Filters (if required) is preferably chosen in such a manner that the OD of each ND Filter is the same and the sum of the ODs equals the required OD upto a pre-specified level of tolerance 
        
        Arguments:
            qs_list (list[list[complex]]) = List of the sets of quantum state coefficients of the photons emitted by the laser
            basis (numpy.array(list[list[complex]])) = Basis of the quantum states of the photons emitted by the laser
            
        Returned Value:
            NDfilter_photons_net (list[photon]) = List of the photons emitted by the weak laser
        """

        avg_transmittance_reqd = 1/self.mu_photons
        avg_OD_reqd = np.round(np.log10(1/avg_transmittance_reqd),decimals = 1)

        if (avg_OD_reqd >= self.min_OD) and (avg_OD_reqd <= self.max_OD):
            NDFilter.set_OD(self,avg_OD_reqd)
            self.ND_Filter_Stack.append(avg_OD_reqd)
            print('OD Stack: ')
            print(self.ND_Filter_Stack)
            self.length_tot += self.thickness
            laser_emission_process = self.env.process(Laser.emit(self,qs_list,basis))
            self.env.run()
            laser_photons_net = laser_emission_process.value
            laser_photons_net = list(deepflatten(laser_photons_net))
            NDfilter_photons_net = NDFilter.attenuate(self,laser_photons_net)
            return NDfilter_photons_net

        elif avg_OD_reqd > self.max_OD:

            OD_base = 0
            OD_multiplier = 0
            tol = 0.1

            search_flag = 0
            searched_OD_stack = {}

            for i in np.round(np.arange(self.min_OD,self.max_OD + 0.1,0.1),1):
                for j in range(2,self.max_num_of_ND_filters + 1,1):
                    if round(abs((i*j) - avg_OD_reqd),1) < tol:
                        searched_OD_stack[i] = j
                        search_flag = 1
                
            print(searched_OD_stack)
            print(f'Search Flag = {search_flag}')
            
            if search_flag == 1:
                OD_base = min(searched_OD_stack,key = searched_OD_stack.get)
                OD_multiplier = searched_OD_stack[OD_base]
                for l in range(1,OD_multiplier+1,1):
                    self.ND_Filter_Stack.append(OD_base)
                print('OD Stack: ')
                print(self.ND_Filter_Stack)

        
            if search_flag != 1:

                num_of_ND_filters = math.ceil(avg_OD_reqd/self.max_OD)
                
                if num_of_ND_filters <= self.max_num_of_ND_filters:
                    
                    net_OD_combinations_list = list(it.combinations_with_replacement(np.round(np.arange(self.min_OD,self.max_OD + 0.1,0.1),1),num_of_ND_filters))
                    
                    summed_net_ODs = np.round(list(map(sum,net_OD_combinations_list)),1)

                    tol_flag = 0
                    
                    print(f'avg_OD_reqd = {avg_OD_reqd}')

                    for sumv in summed_net_ODs:
                        if round(abs(round(sumv,1) - avg_OD_reqd),1) < 0.1:
                            tol_flag = 1
                            idx_reqd = np.min(np.where(summed_net_ODs == sumv))
                            print(idx_reqd)
                            self.ND_Filter_Stack = list(net_OD_combinations_list[idx_reqd])
                            print('OD Stack: ')
                            print(self.ND_Filter_Stack)
                            break
                    if tol_flag == 0:
                        for sumv in summed_net_ODs:
                            if round(abs(round(sumv,1) - avg_OD_reqd)) < 0.2:
                                tol_flag = 2
                                idx_reqd = np.min(np.where(summed_net_ODs == sumv))
                                self.ND_Filter_Stack = list(net_OD_combinations_list[idx_reqd])
                                print('OD Stack: ')
                                print(self.ND_Filter_Stack)
                                break
                    if tol_flag == 0:
                        for sumv in summed_net_ODs:
                            if round(abs(round(sumv,1) - avg_OD_reqd)) < 0.3:
                                tol_flag = 3
                                idx_reqd = np.min(np.where(summed_net_ODs == sumv))
                                self.ND_Filter_Stack = list(net_OD_combinations_list[idx_reqd])
                                print('OD Stack: ')
                                print(self.ND_Filter_Stack)
                                break

                else:         
                    print(f'ERROR: NOT possible to attenuate to ~ single photon levels...use more than {self.max_num_of_ND_filters} ND Filters!') 
                    return None

            self.length_tot += len(self.ND_Filter_Stack)*self.thickness
            laser_emission_process = self.env.process(Laser.emit(self,qs_list,basis))
            self.env.run()
            laser_photons_net = laser_emission_process.value
            laser_photons_net = list(deepflatten(laser_photons_net))
            print(f'\nTotal no. of photons emitted by the laser: {len(laser_photons_net)}\n')
            for OD_val in self.ND_Filter_Stack:
                NDFilter.set_OD(self,OD_val)
                NDfilter_photons_net = NDFilter.attenuate(self,laser_photons_net)
                laser_photons_net = NDfilter_photons_net

            return NDfilter_photons_net




                


