#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import random
import pdb
from copy import copy

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
	    vector = dataset[i]
	    if (vector[-1] not in separated):
		    separated[vector[-1]] = []
	    separated[vector[-1]].append(vector)
    return separated

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector[0])
	return separated


def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
    	summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
    	probabilities[classValue] = 1
    	for i in range(len(classSummaries)):
    		mean, stdev = classSummaries[i]
    		x = inputVector[i]
    		probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
    	if bestLabel is None or probability > bestProb:
    		bestProb = probability
    		bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
    	result = predict(summaries, testSet[i])
    	predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
    	if testSet[x][-1] == predictions[x]:
    		correct += 1
    return (correct/float(len(testSet))) * 100.0
    

def loadData(filename):
    data_matrix = []
    f = open(filename, 'r')
    for line in f:
        tokens = line.split(',')
        data = []
        label = ''
        for i in range(0, len(tokens) - 1):
            data.append(float(tokens[i]))
            label = tokens[-1]
        data.append(label)
        data_matrix.append(data)
    return data_matrix

def loadDataCovertype(filename):
    data_matrix = []
    f = open(filename, 'r')
    aux = 0
    for line in f:
        tokens = line.split(',')
        features = []
        for i in range(10):
            features.append(float(tokens[i]))

        cat_feature = []
        for i in range(10, 14):
            cat_feature.append(float(tokens[i]))
        features.append(cat_feature)

        cat_feature2 = []
        for i in range(14, 54):
            cat_feature2.append(float(tokens[i]))
        features.append(cat_feature2)

        data = (features, tokens[len(tokens) - 1])

        data_matrix.append(data)    
    return data_matrix

def main():
    all_set = loadData('iris/iris.data')
    random.shuffle(all_set)

    training_set = all_set[:math.floor(len(all_set)*0.8)]
    test_set = all_set[math.floor(len(all_set)*0.8):]

    summaries = summarizeByClass(training_set)
    # test model
    predictions = getPredictions(summaries, test_set)
    accuracy = getAccuracy(test_set, predictions)
    print('Accuracy: {0}%').format(accuracy)


main()
