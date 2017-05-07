import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Read in test and train
test = pd.read_table("test.txt", header=None)
train = pd.read_table("train.txt", header=None)

# Rename
file1 = train
file2 = test

# Subset first 9,000 from train and first 1,000 from test
file1 = file1.ix[:8999,:]
file2 = file2.ix[:999,:]

# Concatenate the two files together and reset indexes
new = pd.concat([file1, file2], axis=0)
new = new.reset_index(drop=True)

# Find out how many duplicates are in file
sum(new.ix[:,0]) # ~3700 duplicates

# Below is a manual cross validation setup
train1 = new.ix[0:7999,:]
train2 = pd.concat([new.ix[0:5999,:], new.ix[8000:,:]], axis=0)
train3 = pd.concat([new.ix[0:3999,:], new.ix[6000:,:]], axis=0)
train4 = pd.concat([new.ix[0:1999,:], new.ix[4000:,:]], axis=0)
train5 = new.ix[2000:,:]
    
test1 = new.ix[8000:,:]
test2 = new.ix[6000:7999,:]
test3 = new.ix[4000:5999,:]
test4 = new.ix[2000:3999,:]
test5 = new.ix[0:1999,:]

# Output files
train1.to_csv('train1.txt', index=False, sep='\t')  
train2.to_csv('train2.txt', index=False, sep='\t')  
train3.to_csv('train3.txt', index=False, sep='\t')  
train4.to_csv('train4.txt', index=False, sep='\t')  
train5.to_csv('train5.txt', index=False, sep='\t')  

test1.to_csv('test1.txt', index=False, sep='\t')   
test2.to_csv('test2.txt', index=False, sep='\t')   
test3.to_csv('test3.txt', index=False, sep='\t')   
test4.to_csv('test4.txt', index=False, sep='\t')   
test5.to_csv('test5.txt', index=False, sep='\t')   
