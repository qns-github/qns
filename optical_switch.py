# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""

import numpy as np
import simpy
from component import *
from photon import *
from photon_enc import *
from quantum_state import *

class OpticalSwitch():
    
    """
    Models an Optical Switch
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator
        on_time (float) = Time at which the optical switch is ON
        latency (float) = Extra Time Delay in switching from ON to OFF or OFF to ON
        switching_freq (float) = Frequency of switching from ON to OFF or OFF to ON
    """
    
    def __init__(self,uID,env,on_time,latency,switching_freq):
        
        """
        Constructor for the OpticalSwitch class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
            gen (numpy.random.Generator) = Random Number Generator
            initial_on_time (float) = Time at which the optical switch is ON
            latency (float) = Extra Time Delay in switching from ON to OFF or OFF to ON
            switching_freq (float) = Frequency of switching from ON to OFF or OFF to ON
        """
        
        Component.__init__(self,uID,env)
        self.on_time = on_time
        self.latency = latency
        self.switching_freq = switching_freq
        
    def transmit(self,p,receivers):
        
        """
        Instance method to transmit an incoming photon to one of the 2 attached receivers
        
        Arguments:
            p (photon) = Incoming photon
            receivers (List[QuantumChannel] or List[Detector]) = Receivers attached to the optical switch to receive the incoming photon                                                                                              
        """
        
        # Forward the photon to the 1st receiver if it arrives at the switch when it is ON (including the latency period during the ON state); Otherwise, forward it to the 2nd receiver
        if (self.env.now >= self.on_time) and (self.env.now <= self.on_time + self.latency):
            receivers[0].receive(p)
            self.on_time = self.on_time + (2*self.latency) + (1/self.switching_freq)# The factor of 2 accounts for the latency during both the ON and OFF states of the switch before it goes in the ON state again  
        else:
            receivers[1].receive(p)
                
            
        
        