# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import pytest
import numpy as np
import simpy
from photon import *
from detector import *
from iteration_utilities import deepflatten


UID = 'D1'
DEAD_TIME = 1e-8
DARK_COUNT_RATE = 0
JITTER = 55e-12
MEAN_RESPONSE_TIME = 0

UID_P = 'p1'
WL = 2e-9
TWIDTH = 0.0
ENC_TYPE = 'Polarization'
COEFFS = [[complex(1/2)],[complex(np.sqrt(3)/2)]]
COEFFS = np.reshape(np.array(COEFFS),(len(COEFFS),1))
BASIS = encoding['Polarization'][0]
MBASIS = encoding['Polarization'][0]

def test_init():
    ENV = simpy.Environment()
    XI = 5
    COUPLING_EFFICIENCY = 0.54
    D1 = Detector(UID,ENV,DEAD_TIME,XI,COUPLING_EFFICIENCY,DARK_COUNT_RATE,JITTER,MEAN_RESPONSE_TIME)
    assert D1.uID == UID
    assert D1.dead_time == DEAD_TIME
    assert D1.coupling_eff == COUPLING_EFFICIENCY
    assert D1.dark_count_rate == DARK_COUNT_RATE
    assert D1.jitter == JITTER
    assert D1.mean_response_time == MEAN_RESPONSE_TIME

def test_det_eff():
    ENV = simpy.Environment()
    XI = 6
    COUPLING_EFFICIENCY = 1.0
    D1 = Detector(UID,ENV,DEAD_TIME,XI,COUPLING_EFFICIENCY,DARK_COUNT_RATE,JITTER,MEAN_RESPONSE_TIME)
    for i in range(10000):
        p = photon(UID_P+str(i),WL,TWIDTH,ENC_TYPE,COEFFS,BASIS)
        p_det_process = D1.receive(p,MBASIS,None)
        L = 1000
        c = 3e8
        n_core = 1.5
        transmission_time = L/(c/n_core)
        ENV.timeout(transmission_time)
        ENV.run()
    assert abs((D1.photon_count/10000) - D1.det_eff) < 5e-2
    
def test_coupling_eff():
    ENV = simpy.Environment()
    XI = 8
    COUPLING_EFFICIENCY = 0.6
    D1 = Detector(UID,ENV,DEAD_TIME,XI,COUPLING_EFFICIENCY,DARK_COUNT_RATE,JITTER,MEAN_RESPONSE_TIME)
    for i in range(10000):
        p = photon(UID_P+str(i),WL,TWIDTH,ENC_TYPE,COEFFS,BASIS)
        p_det_process = D1.receive(p,MBASIS,None)
        L = 1000
        c = 3e8
        n_core = 1.5
        transmission_time = L/(c/n_core)
        ENV.timeout(transmission_time)
        ENV.run()
    assert abs((D1.photon_count/10000) - D1.coupling_eff) < 5e-2
   
