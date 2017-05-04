import os
import pandas as pd
import numpy as np
import re, math
from collections import Counter

path = "/Users/jordanbaker/msds_paraphrase_identification"
os.chdir(path)

train = pd.read_table("train.txt", header=None)
test = pd.read_table("test.txt", header=None)

train = train.dropna()
test = test.dropna()

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)
 
acc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for j in acc:
    trainer = pd.DataFrame({'Text1': train.ix[:,1], 'Text2': train.ix[:,2], 'Label': train.ix[:,0]})
    cos_list = []
    label_list = []
    
    for i in range(0,len(train)):
        text1 = train.ix[i,1]
        text2 = train.ix[i,2]
    
        vector1 = text_to_vector(text1)
        vector2 = text_to_vector(text2)
    
        cosine = get_cosine(vector1, vector2)
        cos_list.append(cosine)
        #print('Cosine:', cosine)
        
        if (cosine > j):
            temp_label = 1
        else:
            temp_label = 0
        
        label_list.append(temp_label)
    
    trainer['Cosine'] = cos_list
    trainer['Classifier'] = label_list
    
    accuracy = sum(trainer.Classifier[trainer.Classifier == trainer.Label])/len(trainer)
    print(accuracy)

# 0.1 threshold = 37%
# 0.2 threshold = 37%
# 0.3 threshold = 36%
# 0.4 threshold = 33%
# 0.5 threshold = 29%
# 0.6 threshold = 22%
# 0.7 threshold = 16%
# 0.8 threshold = 9%
# 0.9 threshold = 3%


tester = pd.DataFrame({'Text1': test.ix[:,1], 'Text2': test.ix[:,2], 'Label': test.ix[:,0]})
cos_list = []
label_list = []

for i in range(0,len(test)):
    text1 = test.ix[i,1]
    text2 = test.ix[i,2]

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine = get_cosine(vector1, vector2)
    cos_list.append(cosine)
    print('Cosine:', cosine)
    
    if (cosine > 0.1):
        temp_label = 1
    else:
        temp_label = 0
    
    label_list.append(temp_label)

tester['Cosine'] = cos_list
tester['Classifier'] = label_list

accuracy = sum(tester.Classifier[tester.Classifier == tester.Label])/len(tester)
print(accuracy) # accuracy = 37.14%

