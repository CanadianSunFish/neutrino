from collections import OrderedDict
import numpy as np

class Node:

    def __init__(self, x, y, z) -> None:
        
        self.x = x
        self.y = y
        self.z = z
        
    def get_modulus(self, other):

        return (np.sqrt((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2))
    

node = Node(1, 2, 3)
node1 = Node(2, 3, 4)
node2 = Node(4, 5, 6)
node3 = Node(1, 2, 3)
node4 = Node(1, 2, 3)

mod1 = node.get_modulus(node1)
mod2 = node.get_modulus(node2)
mod3 = node.get_modulus(node3)
mod4 = node.get_modulus(node4)

edges = OrderedDict()

edges[node1] = mod1
edges[node2] = mod2
edges[node3] = mod3

print(1 < max(edges.values()))

edges.pop(list(edges.items())[(list(edges.values()).index(max(edges.values())))][0])

print(edges)

edges[node4] = mod4

print(len(edges))

