# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import numpy as np
import simpy
from laser import *
from photon import *
from photon_enc import *
from quantum_state import *
from component import *
from iteration_utilities import deepflatten

class QuantumChannel(Component):
    
    """
    Models a Quantum Channel (Fiber for the transmission of Quantum Information)
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator
        length (float) = Length 
        alpha (float) = Attenuation Coefficient
        n_core (float) = Refractive Index of the Core
        n_cladding (float) = Refractive Index of the Cladding
        core_dia (float) = Diameter of the Core
        pol_fidelity (float) = Polarization Fidelity (Probability of not undergoing Depolarization)
        chr_dispersion (float) = Chromatic Dispersion 
        depol_prob (float) = Probability of suffering depolarization
        source (node) = Source Node 
        destination (node) = Destination Node
        coupling_eff (float) = Coupling Efficiency of the Source with the Quantum Channel
    """

    def __init__(self,uID,env,length,alpha,n_core,n_cladding,core_dia,pol_fidelity,chr_dispersion,depol_prob):
        
        """
        Constructor for the QuantumChannel class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
            length (float) = Length 
            alpha (float) = Attenuation Coefficient
            n_core (float) = Refractive Index of the Core
            n_cladding (float) = Refractive Index of the Cladding
            core_dia (float) = Diameter of the Core
            pol_fidelity (float) = Polarization Fidelity (Probability of not undergoing Depolarization)
            chr_dispersion (float) = Chromatic Dispersion
            depol_prob (float) = Probability of suffering depolarization
        """
        
        Component.__init__(self,uID,env)
        self.length = length                  
        self.alpha = alpha                  
        self.n_core = n_core               
        self.n_cladding = n_cladding          
        self.core_dia = core_dia              
        self.pol_fidelity = pol_fidelity      
        self.chr_dispersion = chr_dispersion  
        self.depol_prob = depol_prob

    def connect_channel(self,source,destination):
        
        """
        Instance method to connect a quantum channel with its source and destination nodes
        
        Arguments:
            source (node) = Source Node 
            destination (node) = Destination Node
        """
        
        self.source = source
        self.destination = destination
        source.link_qch(self,destination)

    def coupling_efficiency(self,source):
        
        """
        Instance method to compute the coupling efficiency of a quantum channel
        
        Reference : http://www-eng.lbl.gov/~shuman/NEXT/CURRENT_DESIGN/TP/FO/fiber_coupling_efficiency_doric_lenses.pdf
        
        Argument:
            source(Laser or Weaklaser or EntangledPhotonsSourceSPDC) = Source of Photons encoded with Quantum Information
        """
        
        NA = np.sqrt(np.square(self.n_core) - np.square(self.n_cladding))

        if (self.core_dia > source.height) and (self.core_dia > source.length):
            eta_geo = 1 # Geometrical Efficiency
        elif (self.core_dia < source.height) and (self.core_dia < source.length):
            eta_geo = (np.pi()*np.square(self.core_dia))/(4*source.height*source.length) # Geometrical Efficiency
        elif (self.core_dia > source.height) and (self.core_dia < source.length):
            eta_geo = self.core_dia/source.length # Geometrical Efficiency

        R = np.square(self.n_core - 1)/np.square(self.n_core + 1)
        eta_fresnel = 1 - R # Fresnel Efficiency

        m = np.log(0.5)/np.log(np.cos(source.theta_FWHM/2))
        cos_theta_fiber = np.sqrt(1 - np.square(NA))
        eta_angular = 1 - np.power(cos_theta_fiber,m+1) # Angular Efficiency

        self.coupling_eff = eta_geo*eta_fresnel*eta_angular

    def transmit(self,p,source,receiver,receiver_port):
        
        """
        Instance method to transmit the photons emitted by the source
        
        Arguments:
            p (photon) = Photon emitted by the Source
            source(Laser or Weaklaser or EntangledPhotonsSourceSPDC) = Source of Photons encoded with Quantum Information
            
        Returned Value:
            If the photon is successfully transmitted:
                p (photon) = Transmitted photon
            Otherwise:
                None
        """

        rn1 = np.random.rand()
        
        if source is not None:
            self.coupling_efficiency(source)
        else:
            self.coupling_eff = 1.0
        
        trnmt = 10**((-self.alpha*self.length)/10)   # Transmittance
        
        c = 3e8
        # Mean time taken by a photon to cross the length of the quantum channel
        mean_transmission_time = self.length/(c/self.n_core)
        # Calculation of the temporal width of a transmitted photon
        if source is not None:
            p.twidth = self.chr_dispersion*source.lwidth*self.length
        
        # Actual time taken by a photon to cross the length of the quantum channel (considering the effect of chromatic dispersion)
        transmission_time = mean_transmission_time + p.twidth*self.gen.standard_normal()
        
        
        # Check if the probability of the photon being coupled into the fiber is less than the coupling efficiency and if that is the case, couple it into the fiber
        if rn1 < self.coupling_eff:
            rn2 = np.random.rand()
            # Check if the probability of the photon being transmitted by the fiber is less than the transmittance and if that is the case, transmit it
            if rn2 < trnmt:
                rn3 = np.random.rand()
                # Check if the photon uses the polarization encoding scheme of quantum information and if the probability of the photon's polarization remaining unchnaged due to noise is less than the polarization fidelity and if that is the case, return it without corrupting it with noise
                if (p.enc_type == 'Polarization') and (rn3 < self.pol_fidelity):
                    yield self.env.timeout(transmission_time)
                    return p
                else:
                    # Depolarize the photon before transmitting it
                    p.qs.depolarize(self.depol_prob)
                    yield self.env.timeout(transmission_time)
                    return p
            else:
                yield self.env.timeout(transmission_time)
                return None
        else:
            yield self.env.timeout(transmission_time)
            return None

        
    