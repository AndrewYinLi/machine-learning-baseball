import random
import sys
from csv import reader
from operator import itemgetter
from math import log
import operator
import traceback
from collections import defaultdict

attributeIndex={}


class Node(object):
    def __init__(self, value):
        self.value = value
        self.child={}#defaultdict(dict)  #changing this from self.child=defaultdict(dict)

    def print_tree(self):
        return self.preorder_print("")

    def preorder_print(self, traversal):
        if self:
            traversal += (str(self.value) + "-")
            for key in self.child:
            	traversal = self.child[key].preorder_print(traversal)
            # traversal = self.preorder_print(start.right, traversal)
        return traversal

    def isLeaf(self):
    	if len(self.child)==0:
    		return True
    	return False

    # def printLeafNode(self):
    # 	if self:
    # 		for key in self.child:
    # 			if self.child[key].isLeaf:
    # 				return self.child[key]

def getData(fileName,seed,percentage,training,testing):
	#read the file and returning as a list of instances
	with open(fileName, newline='') as f:
		data= list(reader(f, delimiter=','))

	random.seed=seed
	#get the attributes


	attributes=[]
	attributes=getAttr(data,attributes)
	for i in range(len(attributes)):
		attributeIndex[attributes[i]]=i+1

	#remove the first row containing the name of the labels and attributes
	data=data[1:]
	random.shuffle(data)
	# if fileName=="opticalDigit.csv":
	# 	for x in range(len(data)):
	# 		for y in range(1,len(data[x])):
	# 			data[x][y] = float(data[x][y])
	#setting the length of the training/testing
	lenTrain=len(data)
	lenTrain*=percentage
	#lenTest=len(data)-lenTrSet
	training=data[:int(lenTrain)]
	#making sure no data getting lost
	for i in range(len(data)):
		if data[i] not in training:
			testing.append(data[i])
	return training,testing,attributes

def getLabels(dSet):
	labels=[]
	for i in range(len(dSet)):
		if dSet[i][0] not in labels:
			labels.append(dSet[i][0])
	return labels

def getLabelsCount(dSet,labels):
	labelsData={}
	for i in range(len(labels)):
		labelsData[labels[i]]=0
	for i in range(len(dSet)):
		labelsData[dSet[i][0]]+=1
	return labelsData

def getAttr(data,attributes):
	attrData=data[0]
	for j in range(1,len(attrData)):
		attributes.append(attrData[j])
	#print(attributes)
	return attributes

def getAttrValuesPerCol(dSet,attrValues,j):
	attrValues=[]
	for i in range(1,len(dSet)):
		if dSet[i][j] not in attrValues:
			attrValues.append(dSet[i][j])
	return attrValues

def getEntropy(dSet,labelsData):
	entropy=0.0
	for label in labelsData:
		if len(dSet) != 0:
			p = float(labelsData[label]) / len(dSet)
			if p>0:
				entropy -= p * log(p, 2)
	#print("this is entropy",entropy)
	return entropy

def getInformationGain(entropy,subEntropy):
	gain=0.0
	gain=entropy - subEntropy
	return gain

#find a*
def getBestAttr(dSet,labelsData,attributes,aValues):
	entropy=getEntropy(dSet,labelsData)
	maxGain=-1
	aStar=""
	for i in range (len(attributes)):
		attrValues=[]
		attrValues=aValues[attributes[i]]
		#print("this is the values",attrValues)
		subEntropy=0.0
		for value in attrValues:
			subDSet=getPartition(dSet,value,attributes[i])
			p = len(subDSet) / float(len(dSet))
			#print("this is p",p)
			#print(subDSet)
			labelsData=getLabelsCount(subDSet,list(labelsData.keys()))
			subEntropy += p * getEntropy(subDSet,labelsData)
			#print("this is subEntropy",subEntropy)
		gain=getInformationGain(entropy,subEntropy)
		#print("the gain is",gain)
		if (gain > maxGain):  
			maxGain = gain 
			#print(attributes)
			aStar = attributes[i]
		#elif gain<-1:
	# 		print(entropy,subEntropy)
	# 		print("this is attrivbute,",attributes[i])
	# 		print(len(dSet))
	# print("aStar is",aStar)
	return aStar

def getPartition(dSet,value,attributeName):
	newDSet=[]
	index=attributeIndex[attributeName]
	for i in dSet:
			#if i[k]== value:
		if i[index]==value:
			#part = i[:k]  
			#part.extend(i[k + 1:])
			newDSet.append(i)
	return newDSet            	

# def getaStarIndex(attributes,aStar):
# 	for i in range (len(attributes)):
# 		if attributes[i]==aStar:
# 			return i+1

def removeaStar(newAttributes,attributes,aStar):
	for i in range (len(attributes)):
		if attributes[i]!=aStar:
			newAttributes.append(attributes[i])
	return newAttributes

def isSameLabel(labelsData):
	count=0
	for key, value in labelsData.items():
		#print(key,value)
		if value!=0:
			count+=1
	if count==1:
		return True
	else:
		return False

def id3(attributes,s,labels,aValues):
	N=Node("null")
	labelsData=getLabelsCount(s,labels)
	#print((isSameLabel(labelsData)))
	if len(attributes)==0:
		N.value= (max(labelsData.items(), key=operator.itemgetter(1))[0]) #return most common label 
	elif (isSameLabel(labelsData)):
		N.value=(max(labelsData.items(), key=operator.itemgetter(1))[0]) #that would return the only label anyways, the one that wont be 0
	else:
		aStar=getBestAttr(s,labelsData,attributes,aValues)
		N.value=aStar
		v=[]
		v=aValues[aStar]
		for value in v:
			#print(value)
			Sv=getPartition(s,value,aStar)
			#print(value, len(Sv))
			if len(Sv)==0:
				child=Node((max(labelsData.items(), key=operator.itemgetter(1))[0]))
				N.child[value]=child
			else:
				newAttr=[]
				newAttr=removeaStar(newAttr,attributes,aStar)
				newAttr=list(newAttr)
				N.child[value]=id3(newAttr,Sv,labels,aValues)
	return N

def getAccuracy(testing, predictions,match):
    for i in range(len(testing)):
        if testing[i][0] == predictions[i]:
            match += 1
    accur=(match/float(len(testing))) * 100.0
    return accur

def predict(node,instance):
	if node.isLeaf():
		return node.value
	else:
		index=attributeIndex[node.value]
		value=instance[index]
		#print(node.value,value)
		child=node.child[value]
		return predict(child,instance)
	

def main():
	print("This is my id3 program")
	dataFileName=sys.argv[1]
	instancePer=eval(sys.argv[2])
	randSeed=eval(sys.argv[3])
	trainingData=[]
	testingData=[]
	predictions=[]
	predictCount=0
	trainingData,testingData,attributes=getData(dataFileName,randSeed,instancePer,trainingData,testingData)
	#this will be used for the conufsion matrix
	outputname="results_ID3_"+dataFileName+"_"+str(randSeed)+".csv"
	labels=getLabels(trainingData)
	labelsData=getLabelsCount(trainingData,labels)
	attrValues=[]
	aValues=defaultdict(list)
	for i in range(len(attributes)):
		for j in range(1,len(trainingData[i])):
			aValues[attributes[i]]=getAttrValuesPerCol(trainingData,attrValues,i+1)
	N=id3(attributes,trainingData,labels,dict(aValues))
	tree=N
	print(tree.print_tree())

	#accuracy = getAccuracy(testingData, predictions,predictCount)
	confusionMatrix=defaultdict(dict)
	for i in range(len(labels)):
		for j in range(len(labels)):
			confusionMatrix[labels[i]][labels[j]]=0
	confusionMatrixDict=dict(confusionMatrix)
	for i in range(len(testingData)):
		result = predict(N,testingData[i])
		predictions.append(result)
		#print('predicted=' + result + 'actual=' + testingData[i][0])
		confusionMatrixDict[testingData[i][0]][result]+=1
	#print(confusionMatrixDict)
	accuracy = getAccuracy(testingData, predictions,predictCount)
	print(accuracy)

	with open(outputname, 'w') as f:
		for i in range(len(labels)):
			f.write(labels[i]+",")
		f.write("\n")
		for i in range(len(labels)):
			for j in range(len(labels)):
				f.write(str(confusionMatrixDict[labels[i]][labels[j]])+",")
			f.write(labels[i]+",")
			f.write("\n")


if __name__ == '__main__':
    main()

#results_ID3_<DataSet>_<Seed>.csv
#running statement:python id3.py monks1.csv 0.75 12345

#make a dictionary to keep track of attributes values: {attribute: v1,v2,v3
#{temperature:[hot,cold,mild]}}
#test entropy function based on slides

#node
#attribute
# label
# isLeaf
# map of children