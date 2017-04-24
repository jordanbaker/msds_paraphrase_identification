import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

path = "/Users/jordanbaker/Documents/School/University of Virginia/Spring 2017/Machine Learning/Final Project"
os.chdir(path)

# reading in all files
# however, we shouldn't need the stack answers file
# we will need the tags file to match up questions with similar tags
quora = pd.read_csv("quora-questions.csv")
stack_q = pd.read_csv("stackoverflow-questions.csv")
stack_a = pd.read_csv("stackoverflow-answers.csv")
stack_t = pd.read_csv("stackoverflow-tags.csv")

quora = quora[~quora.question2.isnull()]

# reference: https://www.kaggle.com/currie32/d/quora/question-pairs-dataset/predicting-similarity-tfidfvectorizer-doc2vec
def review_to_wordlist(review, remove_stopwords=False):
    # Clean the text, with the option to remove stopwords.
    
    # Convert words to lower case and split them
    words = review.lower().split()

   

    # Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    
    review_text = " ".join(words)

    # Clean the text
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", review_text)
    review_text = re.sub(r"\'s", " 's ", review_text)
    review_text = re.sub(r"\'ve", " 've ", review_text)
    review_text = re.sub(r"n\'t", " 't ", review_text)
    review_text = re.sub(r"\'re", " 're ", review_text)
    review_text = re.sub(r"\'d", " 'd ", review_text)
    review_text = re.sub(r"\'ll", " 'll ", review_text)
    review_text = re.sub(r",", " ", review_text)
    review_text = re.sub(r"\.", " ", review_text)
    review_text = re.sub(r"!", " ", review_text)
    review_text = re.sub(r"\(", " ( ", review_text)
    review_text = re.sub(r"\)", " ) ", review_text)
    review_text = re.sub(r"\?", " ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    
    words = review_text.split()
    
    # Shorten words to their stems
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in words]
    
    review_text = " ".join(stemmed_words)
    
    # Return a list of words
    return(review_text)
    
    
def process_questions(question_list, questions, question_list_name):
# function to transform questions and display progress
    for question in questions:
        question_list.append(review_to_wordlist(question))
        if len(question_list) % 10000 == 0:
            progress = len(question_list)/len(quora) * 100
            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))
            
questions1 = [] 
questions2 = []    
process_questions(questions1, quora_clean.question1, "questions1")
process_questions(questions2, quora_clean.question2, "questions2")







