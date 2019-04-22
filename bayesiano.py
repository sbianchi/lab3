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
        separated[vector[-1]].append(vector[:-1])
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def calculateDistribution(categorical_attribute):
    appearences = [0 for i in range(len(categorical_attribute[0]))]
    for instance in categorical_attribute:
        for j in range(len(instance)):
            if instance[j]:
                appearences[j] += 1
    distribution = [appearences[z]/len(categorical_attribute) for z in range(len(appearences))]
    return distribution

def summarize(dataset):
    summaries = []

    attributes = [[] for i in range(len(dataset[0]))]
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            attributes[j].append(dataset[i][j])

    for attribute in attributes:
        if isinstance(attribute[0], list):
            distribution = calculateDistribution(attribute)
            summaries.append(distribution)
        else:
            summaries.append((mean(attribute), stdev(attribute)))
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
            if isinstance(classSummaries[i], list):
                for j in range(len(inputVector[i])):
                    if inputVector[i][j]:
                        probabilities[classValue] *= classSummaries[i][j]
            else:
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
            label = labelToNumber(tokens[-1])
        data.append(label)
        data_matrix.append(data)
    return data_matrix

def labelToNumber(name):
    if name == 'Iris-setosa\n':
        return 0
    elif name == 'Iris-versicolor\n':
        return 1
    elif name == 'Iris-virginica\n':
        return 2
    else:
        pdb.set_trace()

def loadDataCovertype(filename):
    data_matrix = []
    f = open(filename, 'r')
    for line in f:
        tokens = line.split(',')
        features = []
        for i in range(10):
            features.append(float(tokens[i]))

        cat_feature = []
        for i in range(10, 14):
            cat_feature.append(int(tokens[i]))
        features.append(cat_feature)

        cat_feature2 = []
        for i in range(14, 54):
            cat_feature2.append(int(tokens[i]))
        features.append(cat_feature2)
        features.append(int(tokens[len(tokens) - 1]))

        # data = (features, int(tokens[len(tokens) - 1]))

        data_matrix.append(features)
    return data_matrix

def main():
    # all_set = loadData('iris/iris.data')
    all_set = loadDataCovertype('covertype/covtype.data')
    random.shuffle(all_set)

    training_set = all_set[:math.floor(len(all_set)*0.8)]
    test_set = all_set[math.floor(len(all_set)*0.8):]

    summaries = summarizeByClass(training_set)
    # test model
    predictions = getPredictions(summaries, test_set)
    accuracy = getAccuracy(test_set, predictions)
    print('Accuracy: {0}%'.format(accuracy))


main()
