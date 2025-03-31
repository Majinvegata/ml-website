import pandas as pd
import math
import numpy as np
# Load dataset
data = pd.read_csv('decision_tree_dataset.csv')
# Extract features
features = [feat for feat in data.columns if feat != "answer"]
# Define Node class
class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""
# Function to calculate entropy
def entropy(examples):
    pos = 0.0
    neg = 0.0
    for _, row in examples.iterrows():
        if row["answer"] == "yes":
            pos += 1
        else:
            neg += 1
    if pos == 0.0 or neg == 0.0:
        return 0.0
    else:
        p = pos / (pos + neg)
        n = neg / (pos + neg)
        return -(p * math.log2(p) + n * math.log2(n))
# Function to calculate information gain
def info_gain(examples, attr):
    uniq = np.unique(examples[attr])
    gain = entropy(examples)
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = entropy(subdata)
        gain -= (float(len(subdata)) / float(len(examples))) * sub_e
    return gain
# ID3 Algorithm to build decision tree
def ID3(examples, attrs):
    root = Node()
    max_gain = 0
    max_feat = ""
    for feature in attrs:
        gain = info_gain(examples, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = max_feat
    uniq = np.unique(examples[max_feat])
    for u in uniq:
        subdata = examples[examples[max_feat] == u]
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata["answer"])[0]  # Fixed prediction
            root.children.append(newNode)
        else:
            dummyNode = Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)
            child = ID3(subdata, new_attrs)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    
    return root
# Function to print decision tree
def printTree(root: Node, depth=0):
    print("\t" * depth + root.value, end="")
    if root.isLeaf:
        print(" -> ", root.pred)
    print()
    for child in root.children:
        printTree(child, depth + 1)
# Function to classify a new example
def classify(root: Node, new):
    for child in root.children:
        if child.value == new[root.value]:
            if child.isLeaf:
                print("Predicted Label for new example", new, "is:", child.pred)
                return
            else:
                classify(child.children[0], new)
# Run the ID3 algorithm and classify a new example
root = ID3(data, features)
print("Decision Tree is:")
printTree(root)
print("------------------")
# Test with a new example
new_example = {"outlook": "sunny", "temperature": "hot", "humidity": "normal", "wind": "strong"}
classify(root, new_example)
