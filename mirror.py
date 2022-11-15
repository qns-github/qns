# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
from component import *
from photon import *
from photon_enc import *
from quantum_state import *

class Mirror():
    
    """
    Models a Mirror
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator
        reflectivity (float) = Reflectivity of the Mirror
        noise_level (float) = Probability of the Quantum State of the incoming photon being altered because of Noise
        gamma (float) = probability of losing a photon
        lmda (float) =  probability of a photon getting scattered from the system (without any loss of energy)
    """
    
    def __init__(self,uID,env,reflectivity,noise_level,gamma,lmda):
        
        """
        Constructor for the Mirror class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
            reflectivity (float) = Reflectivity of the Mirror
            noise_level (float) = Probability of the Quantum State of the incoming photon being altered because of Noise
            gamma (float) = probability of losing a photon
            lmda (float) =  probability of a photon getting scattered from the system (without any loss of energy)
        """
        
        Component.__init__(self,uID,env)
        self.reflectivity = reflectivity
        self.noise_level = noise_level
        self.gamma = gamma
        self.lmda = lmda
        
    def reflect(self,p):
        
        """
        Instance method to reflect the incoming photon
        
        Argument:
            p (photon) = Incoming photon 
            
        Returned Value:
            If the photon is successfully reflected:
                p (photon) = Reflected photon
            Otherwise:
                None
        """
        
        rn1 = self.gen.random()
        # Check if the probability of the photon getting reflected by the mirror is less than the reflectivity and if that is the case, reflect it
        if rn1 < self.reflectivity:
            rn2 = self.gen.random()
            # Check if the probability of the photon's quantum state being corrupted by dissipative noise is less than the noise level and if that is the case, corrupt its quantum state with dissipative noise
            if rn2 < self.noise_level:
                p.qs.dampen_phase_amplitude(self.gamma,self.lmda)
                return p
            else:
                return p
        else:
            return None
                
                