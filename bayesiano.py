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
    
def bayes(data):
    separate = separateByClass(data)
    val = separate.values()
    print(stdev([1,2,3]))
    return



def loadData(filename):
    data_matrix = []
    f = open(filename, 'r')
    for line in f:
        tokens = line.split(',')
        features = []
        for i in range(0, len(tokens) - 1):
            features.append(float(tokens[i]))
            data = (features, tokens[len(tokens) - 1])
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

    bayes(training_set)


    #print("Errors in training")
    #print(calculateError(training_set, root))
    #print("Errors in test set")
    #print(calculateError(test_set, root))


main()
