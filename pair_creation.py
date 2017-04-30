import os
import pandas as pd
import random

path = "/Users/jordanbaker/Documents/School/University of Virginia/Spring 2017/Machine Learning/Final Project"
os.chdir(path)

questions = pd.read_csv("stackoverflow-questions.csv", encoding='latin-1')
tags = pd.read_csv("stackoverflow-tags.csv", encoding='latin-1')

questions = questions[['Id', 'Title']]

stack_dups = pd.merge(questions, tags, how='left', on='Id')
stack_dups = stack_dups[(stack_dups.Tag == 'osx') | (stack_dups.Tag == 'numpy') | (stack_dups.Tag == 'windows') | 
                   (stack_dups.Tag == 'oracle') | (stack_dups.Tag == 'git') | (stack_dups.Tag == 'matlab') |
                   (stack_dups.Tag == 'apache') | (stack_dups.Tag == 'ssh') | (stack_dups.Tag == 'c#') |
                   (stack_dups.Tag == 'class-variables')]
stack_dups = stack_dups.reset_index(drop=True)
stack_dups.to_csv("stack_dups.csv")

# hand pick duplicates
temp = list(range(1,21))
dups1 = pd.DataFrame({'Id': [8024750, 24587667, 23879139, 411902, 2137394, 17275852, 18861606, 11295936, 15078606, 15901361, 1504724, 26579843, 31526169, 15512560, 30475410, 14935610, 1776290, 15780076, 13969002, 35394049], 'Number': temp})
dups2 = pd.DataFrame({'Id': [32422764, 14152949, 24423338, 26472935, 3310309, 17347938, 16651431, 68645, 18233560, 16234110, 5551269, 15038640, 25927255, 874461, 2133031, 2146031, 1057666, 1696135, 11528078, 37022888], 'Number': temp})

dup_sample1 = pd.merge(questions, dups1, how='right', on='Id')
dup_sample2 = pd.merge(questions, dups2, how='right', on='Id')

stack_dups = pd.merge(dup_sample1, dup_sample2, how='left', on='Number')
stack_dups = stack_dups[['Title_x', 'Title_y']]
stack_dups.columns = ['Question1', 'Question2']
stack_dups['Label'] = 1

num1 = random.sample(range(len(questions)), 80)
num2 = random.sample(range(len(questions)), 80)
sample1 = [questions.ix[i,1] for i in num1]
sample2 = [questions.ix[i,1] for i in num2]
stack_nodups = pd.DataFrame({'Question1': sample1, 'Question2':sample2})
stack_nodups['Label'] = 0

stack = pd.concat([stack_dups, stack_nodups])
stack = stack.reset_index(drop=True)




