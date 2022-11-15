# -*- coding: utf-8 -*-
"""
@author: Anand Choudhary
"""

from component import *

class Node():
    
    """
    Models a Node 
    
    Attributes:
        uID (str) = Unique ID
        env (simpy.Environment) = Simpy Environment for Simulation
        gen (numpy.random.Generator) = Random Number Generator
        cch (Dict[str:ClassicalChannel]) = Dictonary of Classical Channels
        qch (Dict[str:QuantumChannel]) = Dictonary of Quantum Channels
        protocols () = List of Protocols
    """

    def __init__(self,uID,env):
        
        """
        Constructor for the Node class
        
        Arguments:
            uID (str) = Unique ID
            env (simpy.Environment) = Simpy Environment for Simulation
        """
        
        
        Component.__init__(self,uID,env)
        self.cch = {}       # Dictonary of Classical Channels
        self.qch = {}       # Dictionary of Quantum Channels
        self.protocols = [] # List of Protocols
        
    def link_cch(self,cch,Node2_uID):
        """
        Instance method to link the node (node 1) with another node (node 2) via a classical channel
        
        Arguments:
            cch (ClassicalChannel) = Classical Channel
            Node2_uID (str) = Unique ID of Node 2
        """
        self.cch[Node2_uID] = cch

    def link_qch(self,qch,Node2_uID):
        """
        Instance method to link the node (node 1) with another node (node 2) via a quantum channel
        
        Arguments:
            qch (QuantumChannel) = Classical Channel
            Node2_uID (str) = Unique ID of Node 2
        """
        self.qch[Node2_uID] = qch

    