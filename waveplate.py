# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import numpy as np
from component import *
from photon import *
from photon_enc import *
from quantum_state import *

class WavePlate():
    
    """
    Models a General Wave Plate (Can be used as a Half-Wave Plate, Quarter-Wave Plate)
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator
        alpha = Relative Phase Retardation introduced between the Fast and Slow Axes of the Birefringent Uniaxial Crystal
        theta = Angle made by the Fast Axis of the Birefringent Uniaxial Crystal with the Horizontal
    """
    
    def __init__(self,uID,env,alpha,theta):
        
        """
        Constructor for the WavePlate class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
            alpha = Relative Phase Retardation introduced between the Fast and Slow Axes of the Birefringent Uniaxial Crystal
            theta = Angle made by the Fast Axis of the Birefringent Uniaxial Crystal with the Horizontal
        """
        
        Component.__init__(self,uID,env)
        self.alpha = alpha
        self.theta = theta
    
    def transmit(self,p):
        
        """
        Instance method to transmit an incoming photon
        
        Argument:
            p (photon) = Incoming Photon 
        """
        
        global_phase_factor = np.exp(-0.5*complex(0,self.alpha))
        M11 = np.square(np.cos(self.theta)) + np.exp(complex(0,self.alpha))*np.square(np.sin(self.theta))
        M12 = (1 - np.exp(complex(0,self.alpha)))*np.cos(self.theta)*np.sin(self.theta)
        M21 = M12
        M22 = np.square(np.sin(self.theta)) + np.exp(complex(0,self.alpha))*np.square(np.cos(self.theta))
        # Jones Matrix for the Waveplate
        M = global_phase_factor*np.array([[M11,M12],[M21,M22]])
        p.qs.coeffs = np.matmul(M,p.qs.coeffs)