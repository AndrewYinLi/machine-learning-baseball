import sys
import random
import math
import copy
from collections import defaultdict

class Precondition:
	def __init__(self):
		self.attribute = None
		self.value = None
		self.knownRatio = None

	def setAttribute(self, attribute):
		self.attribute = attribute
			
	def setValue(self, value):
		self.value = value

	def setKnownRatio(self, knownRatio):
		self.knownRatio = knownRatio