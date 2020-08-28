# Clickbait anotomy: Identify clickbait with machine learning
 
# Overview
Thisis a project is a part of the Master Thesis "Clickbait anotomy: Identify clickbait with machine learning", for the Research Master in Humanities, specialising in Human Language Technology at the Vrije Universiteit.

This project aims at analysing the linguistic features of clickbait in order to make a distinction between clickbait and non-clickbait and to engineer features for three different machine classifiers Logistic Regression, Random Forest and Support Vector Machine. 

The results of the analysis shows that syntactic and semantic features are importance to detect clickbait headlines. In this project, 100-dimension word embeddings and encoded sequential part-of-speech and dependency tags are used to represent clickbait headlines, while a 100-dimension document embedding model is trained to represent the content of clickbait. The best performance is achieved SVM clasiffier with word embeddings with the results of 0.82 precision and recall.

# Data
The Data for this project is two dataset: the Clickbait Challenge 2017 dataset and clickbait headline dataset from Chakraborty et al (2016). The firt dataset contains clickbait and non-clickbait headlines and contents. The second one only consist of headlines.
## README

Download two datasets from https://www.clickbait-challenge.org/ and https://github.com/bhargaviparanjape/clickbait

Create a directory "Data" in the same directory as the code

Unzip the data files from two links above, and put the data folders in "Data" folder

Run the scripts in this order:

I. For the linguistic analysis of clickbait and non-clickait

Run:   python preprocessing_data.py

       python analyse_data.py

The results of the analysis in pdf format and stored in folder "Figures"

II. For the feature extraction from the two corpora

Run:   python balanced_data.py

       python extract_features.py

The rsults are two models of embeddings stored in folder "Model" and feature vectors for training in folder "Vector"

II. For the training and evaluating of machine learning algorithms

Run:   python classifier.

The results are reports on the performance of each classifier in txt format, stored in "Results"






