# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import numpy as np
from photon import *
from photon_enc import *
from quantum_state import *
from component import *


class NDFilter(Component):
    
    """
    Models a Neutral Density (ND) Filter with Variable Optical Density (OD)
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator 
        min_OD (float) = Minimum Value of the OD that can be set
        max_OD (float) = Maximum Value of the OD that can be set
        thickness (float) = Thickness 
    """

    def __init__(self,uID,env,min_OD,max_OD,thickness):
        
        """
        Constructor for the NDFilter class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
            min_OD (float) = Minimum Value of the OD that can be set
            max_OD (float) = Maximum Value of the OD that can be set
            thickness (float) = Thickness 
        """
        
        Component.__init__(self,uID,env)
        self.min_OD = min_OD
        self.max_OD = max_OD
        self.thickness = thickness #Typically 1 - 3 mm

    def set_OD(self,OD):
        
        """
        Instance method to set the OD of the ND Filter to a particular value between min_OD and max_OD
        
        Argument:
            OD (float) = Optical Density (OD) to which the ND Filter is to be set
        """
        
        assert (OD >= self.min_OD) and (OD <= self.max_OD), f'The ND Filter [{self.uID}] can only be set at that value of optical density which lies between {self.min_OD} and {self.max_OD}'
        self.OD = OD

    def attenuate(self,photons_net):
        
        """
        Instance method to attentuate the output of a laser
        
        Details:
            The energy of the photon(s) transmitted by the ND Filter is less than or equal to the maximum allowable value of the transmission energy of the ND Filter (as determined by its transmittance)
            Additionally, photons are randomly transmitted based on whether their probability of being transmitted is less than the transmittance of the ND Filter or not
            Further, if in case the energy of each and every photon emitted by the laser is greater than the maximum allowable value of the transmission energy of the ND Filter, no photons would be transmitted by the ND Filter 
        
        Argument:
            photons_net (list[photon]) = List of the photons emitted by the laser
            
        Returned Value:
            new_photons_net (list[photon]) = List of the photons transmitted by the ND Filter
        """
        
        h = 6.626e-34
        c = 3e8
        E_net = 0

        for ph in photons_net:
            E_net = E_net + ((h*c)/(ph.wl))

        transmittance = 10**(-1*self.OD)
        E_transmitted = transmittance*E_net

        E_higher = 0
        for p in photons_net:
            E_p = ((h*c)/(p.wl))
            if E_p > E_transmitted:
                E_higher = 1
            else:
                E_higher = 0
                break

        
        if E_higher == 0:
            E_t = 0
            new_photons_net = []
            j = 0
            flag = 1
            while E_t <= E_transmitted:
                for p in photons_net[:]:
                    if E_t < E_transmitted:
                        rnum = np.abs(self.gen.normal(loc = transmittance,scale = 0.1))
                        if rnum < transmittance:
                            E_t = E_t + ((h*c)/(p.wl))
                            if E_t > E_transmitted:
                                if len(new_photons_net) != 0:
                                    flag = 0
                                    break
                                else:
                                    E_t = 0
                                    continue
                            else:
                                new_photons_net.append(p)
                                photons_net.remove(p)
                        else:
                            continue
                    else:
                        break
                    if flag == 0:
                        break
                j = j + 1
            if flag == 0:
                E_t = E_t - ((h*c)/(p.wl))
                
        else:
            print('NONE of the photons have been transmitted since the energy of each and every photon is greater than the maximum allowable transmission energy through the ND Filter')
            new_photons_net = []

        return new_photons_net
