
import numpy as np 
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.cluster import KMeans

## the global function
stemmer = PorterStemmer()




def ClusterModel(data):
	data['headline'] = data['headline'].str.lower()

	pipeline = Pipeline([('stem',PorterStemmer()),('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1)),])
	pipeline = pipeline.fit_transform(data)




def preprocesssing(text):
	tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
	refined_tokens = []
	for token in tokens:
		if re.search(['[a-zA-Z]'],token):
			refined_tokens.append(token)

	stems = [stemmer.stem(t) for t in refined_tokens]





if __name__ == "__main__":
	data = pd.read_csv("news.csv",sep=",")
	stop = stopwords.words('english')
	











	ClusterModel(data)



