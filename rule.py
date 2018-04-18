import sys
import random
import math
import copy
from collections import defaultdict

class Rule:
	def __init__(self, label, preconditions):
		self.label = label
		self.preconditions = preconditions
		self.accuracy = None
		
	def setLabel(self, label):
		self.label = label
			
	def setPreconditions(self, preconditions):
		self.preconditions = preconditions

	def setAccuracy(self, accuracy):
		self.accuracy = accuracy