#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import random
import pdb
from copy import copy

MOST_PROBABLE_LABEL = -1

class Node:

    def __init__(self, data):
        self.children = []
        self.data = data
        self.threshold_indices = -1
        self.threshold = None
        self.leaf = True
        self.pure = True
        self.label = -1
        if len(data) > 0:
            label = data[0][1]
            for i in range(1, len(data)):
                if label != data[i][1]:
                    self.pure = False
                    break
            if self.pure:
                self.label = data[0][1]
        else:
            global MOST_PROBABLE_LABEL
            self.label = MOST_PROBABLE_LABEL

    def setThresholdIndices(self, index):
        self.threshold_indices = index

    def setThreshold(self, val):
        if val != float('inf'):
            self.threshold = val

    def addChild(self, child):
        self.leaf = False
        self.children.append(child)

    def getThreshold(self):
        return self.threshold

    def getThresholdIndices(self):
        return self.threshold_indices

    def getChildren(self):
        return self.children

    def getData(self):
        return self.data

    def getLabel(self):
        return self.label

    def isLeaf(self):
        return self.leaf

    def isPure(self):
        return self.pure

    def thresholdIndexToString(self, threshold_indices):
        return 'attr '+str(threshold_indices)

    def getLabelOrThreshold(self):
        if not self.isLeaf():
            res = str(self.thresholdIndexToString(self.getThresholdIndices()))
            res += ': ' + str(self.threshold)
            return res
        else:
            return 'label: ' + str(self.label)

# calculates the entropy

def calculateEntropy(data):
    count = {}
    entropy = float(0)
    total = len(data)

    for (features, label) in data:
        if label in count:
            count[label] = count[label] + 1
        else:
            count[label] = 1

    for c in count.values():
        prob = c / total
        entropy = entropy - prob * math.log(prob)
    return entropy


# split the data into a left(true branch) and right(false branch) given a dataset, threshold, and feature index

def split(dataset, threshold, feature_index):
    # print("Dividiendo en hijos")
    if threshold == float('inf'):
        children = [ [] for x in range(len(dataset[0][0][feature_index])) ]
        for binary_column in range(len(dataset[0][0][feature_index])):
            for row in dataset:
                if row[0][feature_index][binary_column]:
                    children[binary_column].append(row)
    else:
        children = [[],[]]
        for datapoint in dataset:
            if datapoint[0][feature_index] <= threshold:
                children[0].append(datapoint)
            else:
                children[1].append(datapoint)
    return children

def calculateLowestEntropy(dataset, feature_index):
    # sort = sorted(dataset, key=lambda tup: tup[0][feature_index])
    # print("Calculando la menor entropia")
    best_entropy = float('inf')
    best_thres = float('inf')
    # Check whether is categoric or numeric attr
    if not isinstance(dataset[0][0][feature_index], list):
        for i in range(0, len(dataset)):
            curr_entropy = 0
            curr_thres = dataset[i][0][feature_index]

            curr_children = split(dataset, curr_thres, feature_index)
            for child in curr_children:
                curr_entropy += calculateEntropy(child) * float(len(child)) / float(len(dataset))
            if curr_entropy < best_entropy:
                best_entropy = curr_entropy
                best_thres = curr_thres
                best_children = curr_children
    else:
        best_children = split(dataset, float('inf'), feature_index)
        for child in best_children:
                best_entropy += calculateEntropy(child) * float(len(child)) / float(len(dataset))

    return (best_entropy, best_thres, best_children)

# I want to know what the threshold, and feature index to split by given a dataset

def chooseBestAttribute(dataset,attr):
    # print("Eligiendo mejor atributo")
    best_feature_index = -1
    best_entropy = float('inf')
    best_threshold = float('inf')
    best_children = []

    for i in range(0, len(dataset[0][0])):
        if (attr[i]):
            (entropy, thres, children) = calculateLowestEntropy(dataset, i)
            if entropy < best_entropy:
                best_entropy = entropy
                best_feature_index = i
                best_threshold = thres
                best_children = children

    return (best_entropy, best_threshold, best_feature_index, best_children)


def majorityVote(node):
    labels = [label for (pt, label) in node.getData()]
    choice = max(set(labels), key=labels.count)
    node.label = choice
    return node

def ID3(node,attr):
    if node.isPure():
        return
    else:
        (entropy, threshold, feature_index, children) = chooseBestAttribute(node.getData(),attr)
        if (feature_index == -1):
            majorityVote(node)
        else:
            node.setThreshold(threshold)
            node.setThresholdIndices(feature_index)

            attr[feature_index] = 0

            for child in children:
                # pdb.set_trace()
                child_node = Node(child)
                node.addChild(child_node)
                ID3(child_node,copy(attr))

def calculateErrorMultipleTrees(dataset, root):
    errors = 0
    local_errors = 0
    num_samples = len(dataset)
    for sample in dataset:
        local_errors = 0
        for tree in root:
            label = isSuccessfulEvaluationMultipleTrees(sample, tree[1])
            if sample[1] == label:
                break
            if label != False:
                errors = errors + 1
                break    
            local_errors += 1
            if local_errors == len(root):
                errors = errors + 1
    return float(errors) / float(num_samples)

def isSuccessfulEvaluationMultipleTrees(sample, node):
    features = sample[0]
    if (node.isLeaf()):
        return node.getLabel()
    else:
        children = node.getChildren()
        threshold = node.getThreshold()
        feature_index = node.getThresholdIndices()
        if threshold == None:
            i = 0
            while not features[feature_index][i]:
                i += 1
            return isSuccessfulEvaluationMultipleTrees(sample,children[i])
        else:
            if features[feature_index] <= threshold:
                return isSuccessfulEvaluationMultipleTrees(sample, children[0])
            else:
                return isSuccessfulEvaluationMultipleTrees(sample, children[1])                    

def calculateError(dataset, root):
    errors = 0
    num_samples = len(dataset)
    for sample in dataset:
        if not isSuccessfulEvaluation(sample, root):
            errors = errors + 1
    return float(errors) / float(num_samples)

def isSuccessfulEvaluation(sample, node):
    features = sample[0]
    label = sample[1]
    if (node.isLeaf()):
        if label == node.getLabel():
            return True
        return False
    else:
        children = node.getChildren()
        threshold = node.getThreshold()
        feature_index = node.getThresholdIndices()
        if threshold == None:
            i = 0
            while not features[feature_index][i]:
                i += 1
            return isSuccessfulEvaluation(sample,children[i])
        else:
            if features[feature_index] <= threshold:
                return isSuccessfulEvaluation(sample, children[0])
            else:
                return isSuccessfulEvaluation(sample, children[1])

def humanizedTree(node, level=0):
    ret = "\t"*level+node.getLabelOrThreshold()+"\n"

    for child in node.getChildren():
        ret += humanizedTree(child, level+1)

    return ret

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

def mostProbableLabel(dataset):
    count = {}
    total = len(dataset)
    most_probable_label = -1

    for (features, label) in dataset:
        if label in count:
            count[label] = count[label] + 1
        else:
            count[label] = 1

    for c in count.values():
        if c > most_probable_label:
            most_probable_label = c

    return most_probable_label


def mainA():
    all_set = loadData('iris/iris.data')
    random.shuffle(all_set)

    training_set = all_set[:math.floor(len(all_set)*0.8)]
    global MOST_PROBABLE_LABEL
    MOST_PROBABLE_LABEL = mostProbableLabel(training_set)
    test_set = all_set[math.floor(len(all_set)*0.8):]

    attr = [1]*len(training_set[0][0])

    #pdb.set_trace()
    root = Node(training_set)
    ID3(root,attr)

    print()
    print(humanizedTree(root))

    print("Errors in training")
    print(calculateError(training_set, root))
    print("Errors in test set")
    print(calculateError(test_set, root))

def mainB():
    # all_set = loadDataCovertype('covertype/covtype.data')
    # random.shuffle(all_set)

    # training_set = all_set[:math.floor(len(all_set)*0.8)]
    # global MOST_PROBABLE_LABEL
    # MOST_PROBABLE_LABEL = mostProbableLabel(training_set)
    # test_set = all_set[math.floor(len(all_set)*0.8):]

    
    all_set = loadData('iris/iris.data')
    random.shuffle(all_set)

    training_set = all_set[:math.floor(len(all_set)*0.8)]
    global MOST_PROBABLE_LABEL
    MOST_PROBABLE_LABEL = mostProbableLabel(training_set)
    test_set = all_set[math.floor(len(all_set)*0.8):]

    attr = [1]*len(training_set[0][0])
    errors = [0]*(len(training_set[0][0])-1)

    used = []
    classes = [x[1] for x in training_set if x[1] not in used and (used.append(x[1]) or True)]
    training_set_c = [ [] for x in range(len(classes)) ]
    test_set_c = [ [] for x in range(len(classes)) ]
    root = [ [] for x in range(len(classes)) ]
    i = 0

    for c in classes:
        attr = [1]*len(training_set[0][0])
        for t in training_set:
            if t[1] == c:
                training_set_c[i].append(t)
            else:
                training_set_c[i].append((t[0],False))


        root[i] = Node(training_set_c[i])

        ID3(root[i],attr)

        print('****************************')
        print(c)
        print('****************************')
        errors[i] = calculateError(training_set_c[i],root[i])
        print("Errors in training: ",errors[i])


        for ts in test_set:
            if ts[1] == c:
                test_set_c[i].append(ts)
            else:
                test_set_c[i].append((ts[0],False))
        print()
        print(humanizedTree(root[i]))
        print()
        i += 1

    ordered_root = sorted(zip(errors, root), key=lambda x: x[0])

    print("Errors in test set",calculateErrorMultipleTrees(test_set, ordered_root))
    

def mainCA():
    all_set = loadDataCovertype('covertype/covtype.data')
    random.shuffle(all_set)

    training_set = all_set[:math.floor(len(all_set)*0.8)]
    global MOST_PROBABLE_LABEL
    MOST_PROBABLE_LABEL = mostProbableLabel(training_set)
    test_set = all_set[math.floor(len(all_set)*0.8):]

    attr = [1]*12
    root = Node(training_set)
    ID3(root,attr)

    print()
    print(humanizedTree(root))

    print("Errors in training")
    # print(calculateError(training_set, root))
    print("Errors in test set")
    #print(calculateError(test_set, root))


mainB()
