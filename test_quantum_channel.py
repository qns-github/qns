# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""
import pytest
import numpy as np
import simpy
from photon import *
from quantum_channel import *

#Quantum Channel
UID = 'QC1'
ENV = simpy.Environment()
LENGTH = 1000
ALPHA = 0.2e-3
N_CORE = 1.50
N_CLADDING = 1.47
CORE_DIA = 50e-6
POL_FIDELITY = 0.25
CHR_DISPERSION = 17e-6
DEPOL_PROB = 0.3

#Photon
UID_P = 'p1'
WL = 2e-9
TWIDTH = 0.0
ENC_TYPE = 'Polarization'
COEFFS = [[complex(1/2)],[complex(np.sqrt(3)/2)]]
COEFFS = np.reshape(np.array(COEFFS),(len(COEFFS),1))
BASIS = encoding['Polarization'][0]

def test_init():
    qc1 = QuantumChannel(UID,ENV,LENGTH,ALPHA,N_CORE,N_CLADDING,CORE_DIA,POL_FIDELITY,CHR_DISPERSION,DEPOL_PROB)
    assert qc1.uID == UID
    assert qc1.length == LENGTH
    assert qc1.alpha == ALPHA
    assert qc1.n_core == N_CORE
    assert qc1.n_cladding == N_CLADDING
    assert qc1.core_dia == CORE_DIA
    assert qc1.pol_fidelity == POL_FIDELITY
    assert qc1.chr_dispersion == CHR_DISPERSION
    assert qc1.depol_prob == DEPOL_PROB
    
def test_transmittance():
 
    qc1 = QuantumChannel(UID,ENV,LENGTH,ALPHA,N_CORE,N_CLADDING,CORE_DIA,POL_FIDELITY,CHR_DISPERSION,DEPOL_PROB)
    TRANSMITTANCE = 10**((-ALPHA*LENGTH)/10) 
    p_net_tr = 0
    for i in range(10000):
        p = photon(UID_P+str(i),WL,TWIDTH,ENC_TYPE,COEFFS,BASIS)
        p_tr_process = ENV.process(qc1.transmit(p,None,None,None))
        ENV.run()
        p_tr = p_tr_process.value
        if p_tr is not None:
            p_net_tr += 1
    assert abs((p_net_tr/10000) - TRANSMITTANCE) < 5e-2
    

        

    