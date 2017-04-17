import os
import pandas as pd
import numpy as np

path = "/Users/jordanbaker/Documents/School/University of Virginia/Spring 2017/Machine Learning/Final Project"
os.chdir(path)

# reading in all files
# however, we shouldn't need the stack answers file
# we will need the tags file to match up questions with similar tags
quora = pd.read_csv("quora-questions.csv")
stack_q = pd.read_csv("stackoverflow-questions.csv")
stack_a = pd.read_csv("stackoverflow-answers.csv")
stack_t = pd.read_csv("stackoverflow-tags.csv")

