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
    res = [math.pow(v1[i] - v2[i], 2) for i in range(dim)]        
    return math.sqrt(sum(res))

def getNeighbors(training_set,test_instance,k):
    distances = [(training_set[index],euclideanDistance(test_instance, training_set[index][0])) for index in range(len(training_set))]          
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors    

def mostVotedLabel(dataset):
    count = {}
    total = len(dataset)
    max_votes = -1
    most_voted_label = ''
    for ((features, label),dist) in dataset:
        count[label] =  count.get(label, 0) + 1
        if count[label] >= max_votes: 
            max_votes, most_voted_label = count[label], label 
    return most_voted_label


def isSuccesfull(real_label,neighbors):
    return (real_label == mostVotedLabel(neighbors))     


def mainiris():
    all_set = loadData('iris/iris.data')
    random.shuffle(all_set)

    training_set = all_set[:math.floor(len(all_set)*0.8)]

    test_set = all_set[math.floor(len(all_set)*0.8):]

    test_res = []

    for k in [1,3,7]:
        errors = 0
        for i in range(len(test_set)):
            n = getNeighbors(training_set,test_set[i][0],k)            
            if (not isSuccesfull(test_set[i][1],n)):
                errors += 1    

        print("Errores para k:",k)
        print()
        print(errors/len(test_set)) 
        print()       

def maincovertype():
    all_set = loadData('covertype/covtype.data')
    random.shuffle(all_set)

    training_set = all_set[:math.floor(len(all_set)*0.8)]

    test_set = all_set[math.floor(len(all_set)*0.999):]
    print(len(test_set))

    test_res = []

    for k in [1,3,7]:
        errors = 0
        for i in range(len(test_set)):
            n = getNeighbors(training_set,test_set[i][0],k)            
            if (not isSuccesfull(test_set[i][1],n)):
                errors += 1    

        print("Errores para k:",k)
        print()
        print(errors/len(test_set)) 
        print() 


mainiris()
