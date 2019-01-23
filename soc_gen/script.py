
import csv
import numpy as np 
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier



def Naive_Bayes_model(data):
	tf_idf = TfidfVectorizer(sublinear_tf=True,min_df = 5, norm='l2',encoding='latin-1',ngram_range=(1,3),stop_words='english')
	features = tf_idf.fit_transform(data.Complaint_reason).toarray()
	labels = data.category_id
	#print(features.shape)
	x_train,x_test,y_train,y_test = train_test_split(data['Complaint_reason'],data['Complaint_Status'], random_state = 0)
	text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
	text_clf = text_clf.fit(x_train,y_train)

	parameters = {'vect__ngram_range': [(1, 1),(1,2),(1,3)], 'tfidf__use_idf': (True,False),'clf__alpha': (1e-2,1e-3)}
	gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
	gs_clf = gs_clf.fit(x_train,y_train)

	#print(gs_clf.best_params_)

	predicted = text_clf.predict(x_test)
	print(np.mean(predicted == y_test))	


def SVM_model(data):
	x_train,x_test,y_train,y_test = train_test_split(data['Complaint_reason'],data['Complaint_Status'], random_state = 42)
	text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-2, n_iter=10, random_state=4)),])
	text_clf_svm = text_clf_svm.fit(x_train,y_train)

	parameters = {'vect__ngram_range': [(1, 1),(1,2)], 'tfidf__use_idf': (True,False),'clf__alpha': (1e-2,1e-3)}
	gs_clf_svm = GridSearchCV(text_clf_svm, parameters, n_jobs=-1)
	gs_clf_svm = gs_clf_svm.fit(x_train,y_train)
	#print(gs_clf_svm.best_params_)

	predicted = gs_clf_svm.predict(x_test)
	print(np.mean(predicted == y_test))	
	print(y_test)
	
	return gs_clf_svm


def prediction(model_name,test_data):
	predict_data = model_name.predict(test_data)
	print(predict_data)



if __name__ == "__main__":
	data = pd.read_csv("train.csv",sep=",")
	column = ['Complaint-Status','Complaint-reason','Company-response']
	data = data[column]
	data = data[pd.notnull(data['Complaint-reason'])]
	data = data[pd.notnull(data['Company-response'])]
	data.columns = ['Complaint_Status','Complaint_reason','Complaint_response']
	data['category_id'] = data['Complaint_Status'].factorize()[0]


	category_id_data = data[['Complaint_Status', 'category_id']].drop_duplicates().sort_values('category_id')
	category_to_id = dict(category_id_data.values)
	id_to_category = dict(category_id_data[['category_id', 'Complaint_Status']].values)



	'''
	## for the test data
	test_data = data = pd.read_csv("test.csv",sep=",")
	test_col = ['Complaint-ID','Complaint-reason']
	test_data = test_data[test_col]
	test_data = test_data[pd.notnull(data['Complaint-ID'])]
	test_data = test_data[pd.notnull(data['Complaint-reason'])]
	test_data.columns = ['Complaint_ID','Complaint_reason']
	test_data['category_id'] = test_data['Complaint_reason'].factorize()[0]

	'''

	'''
	category_id_data = test_data[['Complaint_reason', 'category_id']].drop_duplicates().sort_values('category_id')
	category_to_id = dict(category_id_data.values)
	id_to_category = dict(category_id_data[['category_id', 'Complaint_reason']].values)
	'''

	print("For the SVM model :: ")
	model = SVM_model(data)


	#prediction(model,test_data)
	


