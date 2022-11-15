# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import numpy as np
from component import *
from photon import *
from photon_enc import *
from quantum_state import *

class PhaseShifter():
    
    """
    Models a Phase Shifter
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator
        ref_index (float) = Refractive Index of the Phase Shifter
        length (float) = Length
    """
    
    def __init__(self,uID,env,ref_index,length):
        
        """
        Constructor for the PhaseShifter class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
            ref_index (float) = Refractive Index of the Phase Shifter
            length (float) = Length
        """
        
        Component.__init__(self,uID,env)
        self.ref_index = ref_index
        self.length = length
        
    def shift_phase(self,p):
        
        """
        Instance method for performing a phase shift operation on the incoming single photon state
        
        Argument:
            p (photon) = Incoming photon
        """
        
        n_0 = 1
        c = 3e8
        
        delta = ((self.ref_index - n_0)*self.length)/c
        
        phase_shift_factor = np.exp(complex(0,delta))
        
        p.qs.coeffs = np.array([[0],[phase_shift_factor]])*p.qs.coeffs