import sys
import random
import math
import copy
from collections import defaultdict

class Entry:
    def __init__(self, label):
        self.attribute = {}
        self.label = label
        self.weight = 1.0
        
    def setAttributes(self, attributes, values):
        for i in range(0, len(values)):
            if values[i][0].isdigit():
                values[i] = eval(values[i])
            self.attribute[attributes[i]] = values[i]
            
    def setAttribute(self, attribute, value):
        self.attribute[attribute] = value