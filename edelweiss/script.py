'''
Objective is primarily to arrive at a propensity to foreclose and balance transfer 
an existing loan based on lead indicators such as demographics, internal behavior 
and performance on all credit lines; along with the estimated ‘Time to Foreclose’
'''
import loading

import os 
import pandas as pd 
import numpy as np 
import pickle

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression



def Logistic(customer_data,training_data):
	x_train,x_test,y_train,y_test = train_test_split(training_data["AGREEMENTID"],training_data["FORECLOSURE"],random_state=0)





if __name__ == "__main__":
	#preloading data info of the customers
	#loading.loading_customerdata()
	with open("customer_data.pkl","rb") as f:
		data = pickle.load(f)
		columns = ["AGREEMENTID","CUSTOMERID","LOAN_AMT","NET_DISBURSED_AMT","INTEREST_START_DATE","CURRENT_ROI","NET_RECEIVABLE","NET_LTV"]
		data = data[columns]
		for column in data.columns:
			data = data[pd.notnull(data[column])]
		data['category_id'] = data['AGREEMENTID'].factorize()[0]
		data["AGREEMENTID"][0] = 11220001
		data["FORECLOSURE"] = np.nan

	training_data = pd.read_csv("train_foreclosure.csv",sep = ",")
	training_data = training_data[pd.notnull(data["AGREEMENTID"])]
	training_data['category_id'] = training_data['AGREEMENTID'].factorize()[0]
	

	#refined_data
	if os.path.isfile('./refined.pkl') != True :
		count = 0 # refers to the count of the index of the cell from training_data 
		refined_data = data.copy()
		for i in refined_data["AGREEMENTID"]:
			j = training_data["AGREEMENTID"][count]
			if i == j:
				refined_data["FORECLOSURE"][count] = j
				count += 1
			else:
				refined_data[refined_data.AGREEMENTID != i]
			print(i)
		pickle_out = open("refined.pkl","wb")
		pickle.dump(refined_data,pickle_out)
		pickle_out.close()
	else:
		f1 = open("refined.pkl","rb")
		refined_data = pickle.load(f)

	print(refined_data.head(30))

	Logistic(data,training_data)
	#print(len(data["AGREEMENTID"]))

