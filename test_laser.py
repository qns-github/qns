# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import pytest
import numpy as np
from laser import *
from iteration_utilities import deepflatten

ENV = simpy.Environment()
UID = 'L1'
FREQ = 8e7
WL = 1550e-9
LWIDTH = 0.01e-9
TWIDTH = 100e-15
MU_PHOTONS = 5 
ENC_TYPE = 'Polarization'
NOISE_LEVEL = 0
GAMMA = 0.4
LAMBDA = 0.3
THETA_FWHM = 1e-3
HEIGHT = 10e-3
LENGTH = 120e-3

COEFFS = [[complex(1/2),complex(np.sqrt(3)/2)]]
BASIS = encoding['Polarization'][0]

def test_init():
    L1 = Laser(UID,ENV,FREQ,WL,LWIDTH,TWIDTH,MU_PHOTONS,ENC_TYPE,NOISE_LEVEL,GAMMA,LAMBDA,THETA_FWHM,HEIGHT,LENGTH)
    assert L1.uID == UID
    assert L1.freq == FREQ
    assert L1.wl == WL
    assert L1.lwidth == LWIDTH
    assert L1.twidth == TWIDTH
    assert L1.mu_photons == MU_PHOTONS
    assert L1.enc_type == ENC_TYPE
    assert L1.noise_level == NOISE_LEVEL
    assert L1.gamma == GAMMA
    assert L1.lmda == LAMBDA
    assert L1.theta_FWHM == THETA_FWHM
    assert L1.height == HEIGHT
    assert L1.length == LENGTH

def test_emit():
    L1 = Laser(UID,ENV,FREQ,WL,LWIDTH,TWIDTH,MU_PHOTONS,ENC_TYPE,NOISE_LEVEL,GAMMA,LAMBDA,THETA_FWHM,HEIGHT,LENGTH)
    n_photons_total = []
    for i in range(10000):
        L1_emission_process = ENV.process(L1.emit(COEFFS,BASIS))
        ENV.run()
        L1_photons_net = list(deepflatten(L1_emission_process.value))
        n_photons_total.append(len(L1_photons_net))
    act_mu_photons = sum(n_photons_total)/len(n_photons_total)
    assert (act_mu_photons - MU_PHOTONS) < 5e-2


    