import sys
import random
import math
import copy
from collections import defaultdict

class Node:
	def __init__(self):
		self.attribute = None
		self.children = None
		self.label = None
		self.distance = None  #saving the distribution as a dictionary
		self.threshold = None
		self.attributeType = None
		self.sameAttributes = 0
		self.total = 0

	def isLeaf(self):
		if self.label != None:
			return True
		return False
	
	def addLabel(self, label):
		self.label = label

	def setAttribute(self, attribute):
		self.attribute = attribute

	def setThreshold(self, threshold):
		self.threshold = threshold

	def setDistance(self, S, attributeType):  #must add attribute before use

		self.distance = {}
		self.attributeType = attributeType
		if attributeType == "Nominal":
			for entry in S:
				value = entry.attribute[self.attribute]
				if value == "?":
					continue
				if value == self.attribute:
					self.sameAttributes += 1
				self.total += 1
				if value in self.distance.keys():
					self.distance[value] += 1
				else:
					self.distance[value] = 1
			for value in self.distance.keys():
				if self.total == 0:
					break
				self.distance[value] = self.distance[value]/self.total
			
		elif attributeType == "Continuous":
			self.distance[">"] = 0
			self.distance["<="] = 0
			for entry in S:
				value = entry.attribute[self.attribute]
				if value == "?":
					continue
				self.total+=1
				if value > self.threshold:
					self.distance[">"] +=1
				else:
					self.distance["<="] +=1
			for value in self.distance.keys():
				if self.total == 0:
					break
				self.distance[value] = self.distance[value]/self.total

	def addChildren(self, values):
		self.children = {}
		for value in values:
			self.children[value] = Node()

	# Find the child for continuous functions because can't use the value as a key for the children.
	def findChildContinuous(self, value):
		if value <= self.threshold:
			return self.children["<="]
		return self.children[">"]