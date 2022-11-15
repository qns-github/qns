# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import numpy as np
import scipy.linalg
import warnings
from photon_enc import *
from quantum_state import *
warnings.filterwarnings('ignore')

class photon():
    
    """
    Models a Photon
    
    Attributes:
        uID (str) = Unique ID
        wl (float) = Wavelength
        twidth (float) = Temporal Width
        enc_type (str) = Type of Quantum Information Encoding (see 'photon_enc.py') 
        qs (qstate) = Quantum State 
    """

    def __init__(self,uID,wl,twidth,enc_type,coeffs,basis):
        
        """
        Constructor for the photon class
        
        Arguments:
            uID (str) = Unique ID
            wl (float) = Wavelength
            twidth (float) = Temporal Width
            enc_type (str) = Type of Quantum Information Encoding (see 'photon_enc.py') 
            coeffs (list[complex]) = Quantum State Coefficients 
            basis (numpy.array(list[list[complex]])) = Quantum State Basis 
        """
        
        self.uID = uID                 
        self.wl = wl                  
        self.twidth = twidth           
        self.enc_type = enc_type       
        self.qs = qstate(coeffs,basis) 
