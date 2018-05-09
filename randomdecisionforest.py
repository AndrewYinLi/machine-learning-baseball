import sys


def main():
		fileName = sys.argv[1]
		trainingPercentage = eval(sys.argv[2])  #Training Set Percent
		prunePercentage= eval(sys.argv[3])
		#if prunePercentage > 0:
		#	print("Error: pruning percentage is greater than 0!")
		#	quit()
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
					if fileLine[i] != "?":
						if fileLine[i] not in values[attributes[i-1]]:
							values[attributes[i-1]].append(fileLine[i])
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

		root = c45(training, values, attributes)
		if prunePercentage > 0:
			rules = []
			formRules(root, [], rules)
			finalRules = prune(rules, validate)
			#print("Final rules: " + str(finalRules))


		labelsList = list(labels)
		confusionMatrix=defaultdict(dict)
		for i in range(len(labelsList)):
			for j in range(len(labelsList)):
				confusionMatrix[labelsList[i]][labelsList[j]]=0
		confusionMatrixDict=dict(confusionMatrix)
		for entry in test:
			if prunePercentage == 0:
				prediction = predictNoPrune(entry,root,labels)
			else:
				prediction = predict(entry, finalRules, root, labels, values)
			confusionMatrixDict[entry.label][prediction] +=1

		correct = 0
		total = 0
		
		with open(outputname, 'w') as f:
			for i in range(len(labelsList)):
				f.write(labelsList[i]+",")
			f.write("\n")
			for i in range(len(labelsList)):
				for j in range(len(labelsList)):
					f.write(str(confusionMatrixDict[labelsList[i]][labelsList[j]])+",")
					if i == j:
						correct += confusionMatrixDict[labelsList[i]][labelsList[j]]
					total += confusionMatrixDict[labelsList[i]][labelsList[j]]

				f.write(labelsList[i]+",")
				f.write("\n")
		f.close()
		print(str(correct / total))


if __name__ == '__main__':
	main()