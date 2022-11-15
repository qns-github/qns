# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import numpy as np
import simpy

class Component():
    
    """
    Models a component (base class for all the components)
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator 
    """
    
    def __init__(self,uID,env):
        
        """
        Constructor for the Component class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
        """
        
        self.uID = uID
        self.env = env
        self.gen = np.random.default_rng()