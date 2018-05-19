import sys
import random
import math
import copy
from c45 import C45
from collections import defaultdict
from entry import Entry 
from node import Node
from rule import Rule
from precondition import Precondition

def main():
		fileName = sys.argv[1]
		trainingPercentage = eval(sys.argv[2])  #Training Set Percent
		prunePercentage= eval(sys.argv[3])
		seed = eval(sys.argv[4])
		outputname="results_C45_Pruned_"+fileName+"_"+str(seed)+".csv"
		file = open(fileName, "r")
		count = 0
		entries = []
		attributes = []
		values = {}
		labels = []
		
		for fileLine in file:
			fileLine = fileLine.strip()
			fileLine = fileLine.split(",")
			if count == 0:
				count = 1
				for i in range(1, len(fileLine)):
					attributes.append(fileLine[i])
					values[fileLine[i]] = []
			
			else:
				if fileLine[0] not in labels:
					labels.append(fileLine[0])
				entry = Entry(fileLine[0])
				entry.setAttributes(attributes, fileLine[1:len(fileLine)])
				entries.append(entry)
				for i in range(1, len(fileLine)):
					if fileLine[i] != "?": # and not fileLine[i].isspace():
						if fileLine[i] not in values[attributes[i-1]]:
							fileLineObj = fileLine[i]
							if isinstance(fileLineObj,str):
								if len(fileLineObj) > 1:
									if fileLineObj[0] == '-' and str(int(float(fileLineObj[1:]))).isdigit():
										fileLineObj = -1 * eval(fileLineObj[1:])
							
							values[attributes[i-1]].append(fileLineObj)

		file.close()
		random.seed(seed)
		random.shuffle(entries)

		decimalTrainingPercentage = (int) (len(entries) * trainingPercentage) // 1
		if prunePercentage == 0:
			training = entries[0:decimalTrainingPercentage] # Get training set
			test = entries[decimalTrainingPercentage:len(entries)]
		else:
			training = entries[0:decimalTrainingPercentage] # Get training set
			decimalValidationPercentage = (int) (len(entries) * prunePercentage) // 1
			validate = entries[decimalTrainingPercentage:(decimalValidationPercentage + decimalTrainingPercentage)]
			test = entries[(decimalValidationPercentage + decimalTrainingPercentage):len(entries)]

		tree = C45(training, values, attributes, validate, prunePercentage)
		tree.fit(test, labels, outputname)


if __name__ == '__main__':
	main()
