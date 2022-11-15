
# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import numpy as np
import simpy
from iteration_utilities import deepflatten
from photon import *
from photon_enc import *
from quantum_state import *
from laser import *


class EntangledPhotonsSourceSPDC(Laser):
    
    """
    Models a Type-I Spontaneous Parametric Down Conversion (SPDC) based Source which emits entangled photons (photons in entangled quantum states with an intrinsic degree of entanglement [= epsilon])
    
    Reference: A. G. White, D. F. V. James, P. H. Eberhard, and P. G. Kwiat, "Non-maximally entangled states: production, characterization and utilization," Physical Review Letters, vol. 83, no. 16, pp. 3103-3107, 1999
    
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
        sigma (float) = Effective Phase Matching Function Width of the Non-Linear Crystals
        bell_like_state (str)  = Form of the Entangled Quantum State of the SPDC Photons specified in terms of a Bell State 
        chi (float) = Angle of the Pump Laser's Polarization w.r.t. the Vertical Axis
        efficiency (float) = Pair Production Efficiency of the SPDC process (in pairs per photons)
        height_NL_Xtal (float) = Height of the Non-Linear Crystals
        length_NL_Xtal (float) = Length of the Non-Linear Crystals
        height_tot (float) = Height of the Entire Enclosure consisting of a Laser and the Non-Linear Crystals
        length_tot (float) = Length of the Entire Enclosure consisting of a Laser and the Non-Linear Crystals
    """

    def __init__(self,uID,env,freq,wl,lwidth,twidth,mu_photons,enc_type,noise_level,gamma,lmda,theta_FWHM,height,length,sigma,bell_like_state,chi,efficiency,height_NL_Xtal,length_NL_Xtal):
        
        """
        Constructor for the EntangledPhotonsSourceSPDC class
        
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
            sigma (float) = Effective Phase Matching Function Width of the Non-Linear Crystals
            bell_like_state (str)  = Form of the Entangled Quantum State of the SPDC Photons specified in terms of a Bell State 
            chi (float) = Angle of the Pump Laser's Polarization w.r.t. the Vertical Axis
            efficiency (float) = Pair Production Efficiency of the SPDC process (in pairs per photons)
            height_NL_Xtal (float) = Height of the Non-Linear Crystals
            length_NL_Xtal (float) = Length of the Non-Linear Crystals
        """
        
        Laser.__init__(self,uID,env,freq,wl,lwidth,twidth,mu_photons,enc_type,noise_level,gamma,lmda,theta_FWHM,height,length)
        self.uID = uID                   # Unique ID of the Entangled Photons Source based on SPDC
        self.sigma = sigma               # Effective Phase Matching Function Width of the Non-Linear Crystal
        self.bell_like_state = bell_like_state
        self.chi = chi                   # Angle of the Pump Laser's Polarization w.r.t. the Vertical Axis
        self.efficiency = efficiency     # Pair Production Efficiency of SPDC (in pairs per photons)
        if height > height_NL_Xtal:
            self.height_tot = height
        else:
            self.height_tot = height_NL_Xtal # Height of the entire enclosure consisting of a laser and 2 non-linear crystals (Xtal -> 2e-3 m)
        self.length_tot = length + 2*length_NL_Xtal # Length of the entire enclosure consisting of a laser and 2 non-linear crystals (Xtal -> 10e-3 m)
      


    def SPDC_entangled_states(self,p1,p2):
        
        """
        Instance method for setting the quantum states of the photons as the entangled quantum states (with an intrinsic degree of entanglement) specified by the user
        
        Arguments:
            p1 (photon) = First Photon born out of the SPDC process
            p2 (photon) = Second Photon born out of the SPDC process
        """
        
        p1.qs.basis = np.array([np.kron(p1.qs.basis[0],p1.qs.basis[0]),np.kron(p1.qs.basis[0],p1.qs.basis[1]),np.kron(p1.qs.basis[1],p1.qs.basis[0]),np.kron(p1.qs.basis[1],p1.qs.basis[1])])
        p2.qs.basis = p1.qs.basis
        chi_rad = np.radians(self.chi)
        epsilon = np.tan(chi_rad)
        norm_den = np.sqrt(1 + np.square(epsilon))
        if self.bell_like_state == 'phi+':
           p1.qs.coeffs = [complex(1/norm_den),complex(0/norm_den),complex((epsilon*0)/norm_den),complex((epsilon*1)/norm_den)]
           p1.qs.coeffs = np.reshape(np.array(p1.qs.coeffs),(len(p1.qs.coeffs),1))
           p2.qs.coeffs = p1.qs.coeffs
        elif self.bell_like_state == 'phi-':
           p1.qs.coeffs = [complex(1/norm_den),complex(0/norm_den),complex(-1*(epsilon*0)/norm_den),complex(-1*(epsilon*1)/norm_den)]
           p1.qs.coeffs = np.reshape(np.array(p1.qs.coeffs),(len(p1.qs.coeffs),1))
           p2.qs.coeffs = p1.qs.coeffs
        elif self.bell_like_state == 'psi+':
           p1.qs.coeffs = [complex(0/norm_den),complex(1/norm_den),complex((epsilon*1)/norm_den),complex((epsilon*0)/norm_den)]
           p1.qs.coeffs = np.reshape(np.array(p1.qs.coeffs),(len(p1.qs.coeffs),1))
           p2.qs.coeffs = p1.qs.coeffs
        elif self.bell_like_state == 'psi-':
           p1.qs.coeffs = [complex(0/norm_den),complex(1/norm_den),complex(-1*(epsilon*1)/norm_den),complex(-1*(epsilon*0)/norm_den)]
           p1.qs.coeffs = np.reshape(np.array(p1.qs.coeffs),(len(p1.qs.coeffs),1))
           p2.qs.coeffs = p1.qs.coeffs
        else:
            print('ERROR: Typed the wrong bell like state')


    def emit_pp(self,qs_list,basis):
        
        """
        Instance method for emitting entangled photon pairs generated via the SPDC process
        
        Details:
            The number of pump photons that can successfully undergo SPDC to give rise to pairs of photons is decided by the pair production efficiency of the SPDC process
            Additionally, pump photons are randomly successfully down converted to give rise to pairs of photons
        
        Arguments:
            qs_list (list[list[complex]]) = List of the sets of quantum state coefficients of the photons emitted by the laser
            basis (numpy.array(list[list[complex]])) = Basis of the quantum states of the photons emitted by the laser
            
        Returned Value:
            ephotons_net (list[list[list[photon]]]) = List of entangled photons born out of the SPDC process
        """

        ephotons_net = []
        
        laser_emission_process = self.env.process(Laser.emit(self,qs_list,basis))
        self.env.run()
        laser_photons_net = laser_emission_process.value
        laser_photons_net = list(deepflatten(laser_photons_net))
        
        max_no_of_photon_pairs_gen = int(round(self.efficiency*len(laser_photons_net)))
        flag = 1

        for i,ph in enumerate(laser_photons_net):
            rnum = np.abs(self.gen.normal(loc = self.efficiency,scale = 5*self.efficiency))
            no_of_photon_pairs_gen = int(0.5*len(list(deepflatten(ephotons_net))))
            if no_of_photon_pairs_gen <= max_no_of_photon_pairs_gen:
                if rnum < self.efficiency:
                    qs_ephotons = []
                    epp = []
                    ep1 = photon(str(ph.uID) + '_E0',ph.wl*2,0,ph.enc_type,ph.qs.coeffs,ph.qs.basis)
                    ep2 = photon(str(ph.uID) + '_E1',ph.wl*2,0,ph.enc_type,ph.qs.coeffs,ph.qs.basis)
                    self.SPDC_entangled_states(ep1,ep2)
                    epp = [ep1,ep2]
                    qs_ephotons.append(epp)
                    ephotons_net.append(qs_ephotons)
                no_of_photon_pairs_gen = int(0.5*len(list(deepflatten(ephotons_net))))
                if no_of_photon_pairs_gen > max_no_of_photon_pairs_gen:
                    flag = 0
                    break
            else:
                break   
        if flag == 0:
            ephotons_net.remove(qs_ephotons)
                
        return ephotons_net 

