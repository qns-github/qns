# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import numpy as np
import simpy
from component import *

class ClassicalChannel(Component):
    
    """
    Models a Classical Channel (Fiber for the transmission of Classical Information)
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator
        length (float) = Length
        n_core (float) = Refractive Index of the Core 
        source (node) = Source Node 
        destination (node) = Destination Node
    """

    def __init__(self,uID,env,length,n_core):
        
        """
        Constructor for the Classical Channel class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
            length (float) = Length
            n_core (float) = Refractive Index of the Core
        """
        
        Component.__init__(self,uID,env)
        self.length = length # Length
        self.n_core = n_core                  # Refractive Index of the Core
        

    def connect_channel(self,source,destination):
        
        """
        Instance method to connect a classical channel with its source and destination nodes
        
        Arguments:
            source (node) = Source Node 
            destination (node) = Destination Node
        """
        
        self.source = source
        self.destination = destination
        source.link_cch(self,destination)

    def transmit(self,info):
        
        """
        Instance method to transmit the received information 
        
        Details:
            Transmission of information via a classical channel is assumed to be a lossless process
        
        Arguments:
            info (str) = Classical Information to be transmitted
            
        Returned Value:
            info (str) = Classical Information to be transmitted
        """
        
        c = 3e8
        transmission_time = self.length/(c/self.n_core)
        yield self.env.timeout(transmission_time)
        return info
