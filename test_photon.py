# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import pytest
import numpy as np
from photon import *
from photon_enc import *
from quantum_state import *

UID = 'p1'
WL = 2e-9
TWIDTH = 0.0
ENC_TYPE = 'Polarization'
COEFFS = [[complex(1/2)],[complex(np.sqrt(3)/2)]]
COEFFS = np.reshape(np.array(COEFFS),(len(COEFFS),1))
BASIS = encoding['Polarization'][0]


def test_init():
    p1 = photon(UID,WL,TWIDTH,ENC_TYPE,COEFFS,BASIS)
    assert p1.uID == UID
    assert p1.wl == WL
    assert p1.twidth == TWIDTH
    assert p1.enc_type == ENC_TYPE
    assert np.allclose(p1.qs.coeffs,COEFFS)
    assert p1.qs.coeffs.shape == COEFFS.shape
    assert np.allclose(p1.qs.basis,BASIS)
    assert p1.qs.basis.shape == BASIS.shape
    
