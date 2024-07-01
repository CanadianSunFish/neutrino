import time

start = time.time()

import numpy as np

from collections import OrderedDict

class ConvolutionalNeuralNetwork():

    def __init__(self) -> None:
        
        pass


class Node():

    def __init__(
        self, 
        x,
        y,
        z,
    ):
        
        self.x = x
        self.y = y
        self.z = z
        self.edges = OrderedDict()
        self.tensor = None

    def __str__(self) -> str:
        if self.tensor is not None:
            return f"{self.tensor}"
        return f"X: {self.x}\nY: {self.y}\nZ: {self.z}"

    def link_tensor(self, tensor):
        self.tensor = tensor

    def get_position(self):
        return self.x, self.y, self.z
    
    def get_modulus(self, other):
        return (np.sqrt((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2))
    
    def add_edge(self, other):
        mod = self.get_modulus(other)

        if len(self.edges) < 8:
            self.edges[other] = mod
            return

        if mod < max(self.edges.values()):
            self.edges.pop(list(self.edges.items())[(list(self.edges.values()).index(max(self.edges.values())))][0])
            self.edges[other] = mod
            return

        return

class Tensor():

    def __init__(
        self, 
        charge, 
        dom_number, 
        dom_time, 
        hlc, 
        string
    ):
        
        self.charge = charge
        self.dom_number = dom_number
        self.dom_time = dom_time
        self.hlc = hlc
        self.string = string
        self.node = None
    
    def __str__(self) -> str:
        return f"""
X: {self.node.x}
Y: {self.node.y}
Z: {self.node.z}
Charge: {self.charge}
DOM Number: {self.dom_number}
DOM Time: {self.dom_time}
HLC: {self.hlc}
String: {self.string}
"""

    def link_node(self, node):
        self.node = node
        node.link_tensor(self)




if __name__ == "__main__":

    node=Node(1,2,3)
    tensor=Tensor(4, 5, 6, 7, 8)

    tensor.link_node(node)

    

    # from data_process import *

    # pulse, truth = get_db('/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db')

    # print(np.array(pulse))





end = time.time()
print(f"{end-start:.5f}s")
