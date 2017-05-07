# Distributional Question Similarity "Hey, did you already ask that?"

The purpose of this project is to identify whether two questions are semantically linked, in other words, determine if they are "paraphrases" of eachother or not. We employ a variety of methods, some our own and some from other successful iterations of the same taks. The primary method we use is from Ji and Einstein (2013) and their "state of the art" TF-KLD weighting method: http://www.cc.gatech.edu/~jeisenst/papers/ji-emnlp-2013.pdf

We reweighted the doc-term matrix with TF-KLD, performed singular value decompistion to translate the data to latent representation, and used a supervised classification technique called Support Vector Machines (SVM) to classify instances.
More work could be done to tune the cost parameter, epsilon parameter, and K in the singular value decompisition step.

# Data and Hardware
Data was retrieved from the Quora Question Database from Kaggle.com.  
An AWS EC2 and S3 instance was spun up to handle the data: 5 fold cross validation with 8,000 training observations and 2,000 testing observations per fold.

# Extensibility
Once the system was built with the Quora database, we tested out a set of paired stack overflow questions as a proof of concept.  We achieved an Accuracy of 80% over 100 observations.  To further test this system, more questions could be tested, and CV could be used.

# Further Work
Train and test a host of other classification methods including:
-Logistic Regression
-Random Forest and XGBoost
-DL or ANNs
-LDA

# Running the code
Run the python scripts in the following order to reproduce results: <br />
-paraphrase_identification.py = cleaning, manipulation, and structuring of the raw data to the proper format <br />
-weighting.py = perform TF-KLD weighting <br />
-dr.py = perform singular value decompisition <br />
-create.py = creates supervised learning setup with the sample vectors and ground truth labels <br />
-svm.py = performs SVM classification and gets the Accuracy results

You will need to acquire the data from Kaggle and save it in your desired working directory  (store in the same location ideally as the scripts).

# Resources
Kaggle Quora Data: https://www.kaggle.com/quora/question-pairs-dataset
Kaggle StackOverflow Data: https://www.kaggle.com/stackoverflow/pythonquestions
TF-KLD GitHub Page: https://github.com/jiyfeng/tfkld/tree/master/python
ACL Benchmark: https://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_(State_of_the_art)





