##Ali Farahmand
##CS365
##Lab C - DecisionTree

import sys
import math
import copy

file = open(sys.argv[1],"r")
words = []
for line in file:
	words.append(line.split())
attNames = words[0] #list of attribute names
words.pop(0) #remove attribute line from the data


def decision_count(data): #check to see how many yes and no s are in the decision part of training data
    yes = 0
    no = 0
    for row in data:
    	decision = row[-1]
    	if decision == 'yes':
    		yes += 1
    	elif decision == 'no':
    		no += 1
    return yes,no

def allOptions(data, col): #return all unique types for each given attribute column
    return set([row[col] for row in data])

def getNameIndex(attName): #return index of the given attribute
	for i in range(len(attNames)):
		if attNames[i] == attName:
			return i

def condition(attName, attribute, example): #check to see if a condition is met with the given attribute
	index = getNameIndex(attName)
	return example[index] == attribute

def split(data, attName, attribute): #divide the data into positive set and negative set according to condition.
	positive, negative = [], [] 
	for row in data:
		if condition(attName, attribute, row) == True:
			positive.append(row)
		else:
			negative.append(row)
	return positive, negative

def entropy(counts): #calculate the messiness of a single set of data
	if counts[0] == 0 or counts[1] == 0:
		return 0
	else:
		total = counts[0]+counts[1]
		return (-(counts[0]/total*math.log(counts[0]/total,2))-(counts[1]/total)*math.log(counts[1]/total,2))

def entropySplit(data, counts, attName): #calculate the messiness of a set of data given a condition
	total = counts[0]+counts[1]
	result = 0.0
	for option in allOptions(data, getNameIndex(attName)):
		result += (decision_count(split(data, attName, option)[0])[0]+decision_count(split(data, attName, option)[0])[1])/total*entropy(decision_count(split(data, attName, option)[0]))
	return result

def gain(data, attName): #information gain for the given attribute
	ent = entropy(decision_count(data))
	entsplit = entropySplit(data, decision_count(data), attName)
	return(ent-entsplit)

def bestAttr(data): #return the attribute with the highest information gain value
	bestEntropy = 2
	bestGain = 0
	for name in attNames[:-1]:
		curr = gain(data, name)
		#print(curr)
		if curr > bestGain:
			bestGain = curr
			bestName = name
	if bestGain == 0:
		return 0
	return(bestName)

def predict(counts): #predict the goal classification ##in case of duplicate attributes with same goal predicate returns the majority vote ###in case of tie returns zero
	if counts[0]>counts[1]:
		return ('yes')
	else:
		return ('no')

class Leaf: #leaf node constructer
    def __init__(self, data):
        self.prediction = predict(decision_count(data))
class DecisionNode: #decision node constructer
	def __init__(self, attName, option, children):
		self.attName = attName
		self.option = option
		self.children = children

def myTree(root,data): #decision tree learning
    bestName = bestAttr(data)

    if bestName == 0:
        child = Leaf(data)
        root.children.append(child)
        return root

    else:
        children = []

        for option in allOptions(data, getNameIndex(bestName)):

            pos = split(data, bestName, option)[0]

            node = DecisionNode(bestName,option,[])			
            child = myTree(node,pos)
            root.children.append(child)

        return root


def printTree(node,indent=""): #print the tree in a similar format to the given one in the assignment
	if isinstance(node, Leaf):
		print(indent + node.prediction)
		return
	print(indent + node.attName + " : " + str(node.option) + "?")
	for child in node.children:
		printTree(child, indent + "|  ")

def classifier(row_dict, node): #classify the goal using the tree and predict function
	if isinstance(node, Leaf):
		return node.prediction
	if isinstance(node.children[0], Leaf):
		#print("len of children")
		#print(len(node.children))
		return node.children[0].prediction
	attr = node.children[0].attName
	value = row_dict[attr]
	curr = None
	for child in node.children:
		curr = child
		if curr.option == value:
			break
	return classifier(row_dict, curr)

def accuracyTest(words): #leave-one-out cross-validation accuracy test
	a = -1
	data = copy.deepcopy(words)
	dicts = []
	for row in data:
		dict = {}
		for name in attNames[:-1]:
			i = getNameIndex(name)
			dict.update({attNames[i] : row[i]})
		a += 1
		dict.update({"index" : a})
		dict.update({"expected" : row[-1]})
		dicts.append(dict)
	total = len(dicts)
	count = 0
	for item in dicts:
		nwords = copy.deepcopy(words)
		nwords.pop(item['index'])
		root = DecisionNode("root", None, [])
		root = myTree(root, nwords)
		ans = classifier(item,root)
		if ans == item["expected"]:
			count += 1
	print ("guessed correct : " + str(count)+" Out of : " + str(total))
	print ("accuracy : " + str(count/total*100) + "%")

if __name__ == '__main__':
	root = DecisionNode("root",None,[])
	root = myTree(root,words)
	print("Training Data Tree:")
	printTree(root)

	print("======================================")

	print("Accuracy test with leave-one-out cross-validations:")
	accuracyTest(words)

