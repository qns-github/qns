# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
from component import *
from photon import *
from photon_enc import *
from quantum_state import *

class NonPolarisingBeamSplitter():
    
    """
    Models a Non-Polarising Beam Splitter
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator
        theta (float) = Angle of the Beam Splitter (in degrees)
        receivers (List[QuantumChannel] or List[Detector]) = Receivers attached to the beam splitter to receive the reflected/transmitted photon respectively
    """
    
    def __init__(self,uID,env,R):
        
        """
        Constructor for the NonPolarisingBeamSplitter class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
            theta (float) = Angle of the Beam Splitter (in degrees)
        """
        
        Component.__init__(self,uID,env)
        self.theta = np.deg2rad(theta)
        
    def connect(self,receivers):
        """
        Instance method to connect a beamsplitter with its receivers
        
        Arguments:
            receivers (List[QuantumChannel] or List[Detector]) = Receivers attached to the beam splitter to receive the reflected/transmitted photon respectively
        """
        self.receivers = receivers
        
        
    def receive(self,p,input_port):
        
        """
        Instance method to receive and consequently, direct an incoming photon to one of two attached receivers depending upon the reflectance of the beam splitter 
        
        Arguments:
            p (photon) = Incoming photon
            input_port (int) = The input port of the beamsplitter which receives the incoming photon
        """
        
        R_sqr = np.round(np.square(np.cos(self.theta)),1)
        T_sqr = np.round(np.square(np.cos(self.theta)),1)
        rn = self.gen.random()
        
        if input_port == 1:
            if R_sqr >= T_sqr:
                if rn < R_sqr:
                    self.receivers[0].receive(p)
                else:
                    self.receivers[1].receive(p)
            else:
                if rn < T_sqr:
                    self.receivers[1].receive(p)
                else:
                    self.receivers[0].receive(p)
        elif input_port == 2:
            if R_sqr >= T_sqr:
                if rn < R_sqr:
                    self.receivers[1].receive(p)
                else:
                    self.receivers[0].receive(p)
            else:
                if rn < T_sqr:
                    self.receivers[0].receive(p)
                else:
                    self.receivers[1].receive(p)      
        else:
            print('ERROR: Incorrect input port number entered! Please enter either 1 or 2 only')
        
        
   
            