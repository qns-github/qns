# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import numpy as np
import math
from component import *
from photon import *
from photon_enc import *
from quantum_state import *

class Detector():
    
    """
    Models a Single Photon Detector
    
    References: 
        1. R. H. Hadfield, "Single-photon detectors for optical quantum information applications," Nature Photonics, vol. 3, pp. 696–705, 2009
        2. M. Lasota and P. Kolenderski, "Optimal photon pairs for quantum communication protocols," Nature Scientific Reports, vol. 10, no. 20810, 2020
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator
        dead_time (float) = Dead Time
        xi (float) = Multiplicative Factor for deciding the Width of the Time Window for Single Photon Detection
        coupling_eff (float) = Coupling Efficiency 
        dark_count_rate (float) = Rate of generation of Dark Counts
        jitter (float) = Standard Deviation in the Time Interval between the absorption of a Photon and the generation of an Output Electrical Signal from the Detector
        mean_response_time (float) = Average Time taken to generate an Output Electrical Signal after the absorption of a Photon
        det_eff (float) = Detection Efficiency [Probability of registering a Count if a Photon arrives at the Detector]
        next_detection_time (float) = Time at which the next detection event may take place
        next_dark_count_time (float) = Time at which the next dark count is registered 
        detect_dark (bool) = True when a dark count is to be registered; False otherwise
        photon_count (int) = Number of photons which have been successfully detected
        func_call_tracker (int) = Tracks whether the detector's receive function has been called or not, i.e., whether the detector is active or not (A value of zero indicates that the detector is not active)
    """

    def __init__(self,uID,env,dead_time,xi,coupling_eff,dark_count_rate,jitter,mean_response_time):
        
        """
        Constructor for the Detector class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
            dead_time (float) = Dead Time
            xi (float) = Multiplicative Factor for deciding the Width of the Time Window for Single Photon Detection
            coupling_eff (float) = Coupling Efficiency 
            dark_count_rate (float) = Rate of generation of Dark Counts
            jitter (float) = Standard Deviation in the Time Interval between the absorption of a Photon and the generation of an Output Electrical Signal from the Detector
            mean_response_time (float) = Average Time taken to generate an Output Electrical Signal after the absorption of a Photon
        """
        
        Component.__init__(self,uID,env)
        self.dead_time = dead_time
        self.xi = xi
        self.coupling_eff = coupling_eff
        self.dark_count_rate = dark_count_rate
        self.jitter = jitter 
        self.mean_response_time = mean_response_time
        self.det_eff = math.erf(xi/(2*math.sqrt(2)))
        self.next_detection_time = self.env.now
        self.next_dark_count_time = self.env.now
        self.detect_dark = False
        self.photon_count = 0     
        self.func_call_tracker = 0                         

    def schedule_dark_count(self):
        
        """
        Instance method for scheduling a dark count
        
        Details:
            The time intervals for dark counts follow an exponential distribution
        """

        
        dark_count_time_interval = self.gen.exponential(1/self.dark_count_rate)
        self.next_dark_count_time = self.next_dark_count_time + dark_count_time_interval
            

    def receive(self,p,mbasis,follow_up_function = None):
        
        """
        Instance method for receiving a photon for detection and detecting it if the appropriate conditions are satisfied
        
        Arguments:
            p (photon) = Photon for detection
            mbasis (numpy.array(list[list[complex]])) = Measurement Basis 
            follow_up_function (function) = An optionally specified and defined function which maybe called post the detection of a photon 
        """
        
        # Determine the instant(s) of time at which dark count(s) is(are) to be registered 
        if self.dark_count_rate > 0:
            if (self.func_call_tracker == 0) or (self.env.now > self.next_dark_count_time):
                self.schedule_dark_count()
                
            self.func_call_tracker += 1
            
            while (self.env.now > self.next_dark_count_time):
                self.schedule_dark_count()
                self.detect_dark = True
                self.photon_count += 1
            
            # Set detect_dark = True if the current simulation time is approx. equal to the time for registering the next dark count. Otherwise, set detect_dark = False
            if abs(self.env.now - self.next_dark_count_time) <= 1e-10:
                self.detect_dark = True
            else:
                self.detect_dark = False
        
        else:
            self.detect_dark = False
        
        rn1 = self.gen.random()
        # Check if the probability of the photon being coupled into the detector is less than the coupling efficiency and if that is the case, couple it into the detector
        if rn1 < self.coupling_eff:
            # Check if the photon arrives at the detector at/after the next detection time and if at that arrival time, no dark count is scheduled. If that is the case, the photon may be detected.
            if (self.env.now >= self.next_detection_time) and not(self.detect_dark):
                rn2 = self.gen.random()
                # Check if the probability of the photon triggering a detection count is less than the detection efficiency of the detector and if that is the case, register a photon count
                if rn2 < self.det_eff:
                    self.photon_count += 1
                    # Measure the quantum state of the photon
                    self.detect(p,mbasis)
                    # Call any follow up function (if specified as an argument) 
                    if follow_up_function is not None:
                        follow_up_function(p,self.photon_count)
                    # Set the next instant of time at which a photon can possibly be detected (Here, the detector's response function has been assumed to be Gaussian)
                    self.next_detection_time = self.env.now + (self.mean_response_time + self.jitter*self.gen.standard_normal()) + self.dead_time             
        
        elif self.detect_dark:
            # Register a dark count
            self.photon_count += 1
            
        
    def detect(self,p,mbasis):
        
        """
        Instance method for measuring the quantum state of a photon upon its successful detection
        
        p (photon) = Successfully detected Photon
        mbasis (numpy.array(list[list[complex]])) = Measurement Basis
        """

        if p.qs.coeffs.size == 2:
            p.qs.measure_single_qubit_basis(mbasis)
        else:
            p.qs.measure_multiple_qubit_basis_scheme1(mbasis)
    