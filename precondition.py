import sys
import random
import math
import copy
from collections import defaultdict

class Precondition:
	def __init__(self, attribute, value, knownRatio):
		self.attribute = attribute
		self.value = value
		self.knownRatio = knownRatio

	def setAttribute(self, attribute):
		self.attribute = attribute
			
	def setValue(self, value):
		self.value = value

	def setKnownRatio(self, knownRatio):
		self.knownRatio = knownRatio