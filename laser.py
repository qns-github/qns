# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""

import simpy
from photon import *
from photon_enc import *
from component import *
from quantum_state import *

class Laser(Component):
    
    """
    Models a Laser
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator 
        freq (float) = Frequency with which the photons are emitted
        wl (float) = Wavelength
        lwidth (float) = Linewidth 
        twidth (float) = Temporal Width
        mu_photons (int) = Mean Number of Photons
        enc_type (str) = Type of Quantum Information Encoding (see 'photon_enc.py') of the emitted Photons
        noise_level (float) = Probability of the Quantum State of the emitted Photons being altered because of Noise
        gamma (float) = probability of losing a photon
        lmda (float) =  probability of a photon getting scattered from the system (without any loss of energy)
        theta_FWHM (float) = Divergence Angle at Full Width at Half Maximum (FWHM)
        height (float) = Height
        length (float) = Length
    """
    
    def __init__(self,uID,env,freq,wl,lwidth,twidth,mu_photons,enc_type,noise_level,gamma,lmda,theta_FWHM,height,length):
        
        """
        Constructor for the Laser class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation 
            freq (float) = Frequency with which the photons are emitted
            wl (float) = Wavelength
            lwidth (float) = Linewidth 
            twidth (float) = Temporal Width
            mu_photons (int) = Mean Number of Photons
            enc_type (str) = Type of Quantum Information Encoding (see 'photon_enc.py') of the emitted Photons
            noise_level (float) = Probability of the Quantum State of the emitted Photons being altered because of Noise
            gamma (float) = probability of losing a photon
            lmda (float) =  probability of a photon getting scattered from the system (without any loss of energy)
            theta_FWHM (float) = Divergence Angle at Full Width at Half Maximum (FWHM)
            height (float) = Height
            length (float) = Length
        """
        
        Component.__init__(self,uID,env)
        self.freq = freq             
        self.wl = wl                 
        self.lwidth = lwidth         
        self.twidth = twidth         
        self.mu_photons = mu_photons 
        self.enc_type = enc_type     
        self.noise_level = noise_level 
        self.gamma = gamma
        self.lmda = lmda
        self.theta_FWHM = theta_FWHM 
        self.height = height         
        self.length = length         

    def emit(self,qs_list,basis):

        """
        Instance method for emitting photons
        
        Details:
            The number of photons emitted in every time period of the laser are determined by a Poisson distribution with its mean as the mean number of photons (mu_photons)
            Few photons maybe corrupted by a completely dissipative noise: amplitude and phase damping noise depending on the noise level (noise_level) of the laser
        
        Arguments:
            qs_list (list[list[complex]]) = List of the sets of quantum state coefficients
            basis (numpy.array(list[list[complex]])) = Basis of the quantum states
            
        Returned Value:
            photons_net (list[list[photon]]) = List of the emitted photons 
        """
        
        time_pd = 1/self.freq

        photons_net = []

        for i,qs in enumerate(qs_list):
            num_of_photons = self.gen.poisson(lam = self.mu_photons)
            qs_photons = []
            for j in range(num_of_photons):
                wl_p = self.wl + self.lwidth*self.gen.standard_normal()
                p = photon(str(self.uID) + '_' + str(i) + '_' + str(j),wl_p,0,self.enc_type,qs,basis)
                if self.gen.random() < self.noise_level:
                    p.qs.dampen_phase_and_amplitude(self.gamma,self.lmda)
                qs_photons.append(p)
            photons_net.append(qs_photons)
            yield self.env.timeout(time_pd)
       
        return photons_net



