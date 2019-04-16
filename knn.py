#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import random

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
        #aux += 1
        #if aux == 500:
        #   break
    return data_matrix

def euclideanDistance(v1, v2):
    dim, res = len(v1), 0
    for i in range(dim):
        res += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(res)

def getNeighbors(training_set,test_instance,k):

    distances = []
    for index in range(len(training_set)):
        dist = euclideanDistance(test_instance, training_set[index][0])
        distances.append((training_set[index], dist))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors    

def mostVotedLabel(dataset):
    count = {}
    total = len(dataset)
    max_votes = -1
    most_voted_label = ''

    for ((features, label),dist) in dataset:
        if label in count:
            count[label] = count[label] + 1
        else:
            count[label] = 1  

    for key,value in count.items():
        if value > max_votes:
            most_voted_label = key
    return most_voted_label


def isSuccesfull(real_label,neighbors,k):
    return (real_label == mostVotedLabel(neighbors))     


def main():
    all_set = loadData('iris/iris.data')
    random.shuffle(all_set)

    training_set = all_set[:math.floor(len(all_set)*0.1)]

    test_set = all_set[math.floor(len(all_set)*0.8):]

    test_res = []

    for k in [1,3,7]:
        errors = 0
        for i in range(len(test_set)):
            n = getNeighbors(training_set,test_set[i][0],k)            
            if (not isSuccesfull(test_set[i][1],n,k)):
                errors += 1    

        print("Errores para k:")
        print(k)
        print()
        print(errors/len(test_set)) 
        print()       

main()
