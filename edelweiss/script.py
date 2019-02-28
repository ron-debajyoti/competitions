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
from sklearn import metrics



def Logistic(training_data,columns):
	x_train,x_test,y_train,y_test = train_test_split(training_data[columns[:-1]],training_data[columns[-1]],random_state=0,train_size = 0.7)
	multi_logr = LogisticRegression(multi_class = "multinomial",solver="saga").fit(x_train,y_train)
	print(metrics.accuracy_score(y_test,multi_logr.predict(x_test)))
	print("\n")
	return multi_logr



def prediction(model_name,test_data):
	predicted_data = model_name.predict_proba(test_data)[:,1]
	l = predicted_data.tolist()
	predicted_data = np.reshape(predicted_data,(len(l),-1))
	pr_data = pd.DataFrame({"FORECLOSURE":np.array(predicted_data).flatten()})
	return pr_data








if __name__ == "__main__":
	#preloading data info of the customers
	#loading.loading_customerdata()
	with open("customer_data.pkl","rb") as f:
		data = pickle.load(f)
		columns = ["AGREEMENTID","CUSTOMERID","LOAN_AMT","NET_DISBURSED_AMT","CURRENT_ROI","NET_RECEIVABLE","NET_LTV"]
		data = data[columns]
		for column in data.columns:
			data = data[pd.notnull(data[column])]
		data['category_id'] = data['AGREEMENTID'].factorize()[0]
		data["AGREEMENTID"][0] = 11220001
		data["FORECLOSURE"] = 0.0


	# reading training_data and test_data from the respective .csv files
	training_data = pd.read_csv("train_foreclosure.csv",sep = ",")
	training_data = training_data[pd.notnull(data["AGREEMENTID"])]
	test_data = pd.read_csv("test_foreclosure.csv",sep = ",")
	test_data = test_data[pd.notnull(test_data["AGREEMENTID"])]
	
	





	#refined_data
	if os.path.isfile('./refined.pkl') == False :
		count = 0 # refers to the count of the index of the cell from training_data 
		columns = ["AGREEMENTID","CUSTOMERID","LOAN_AMT","NET_DISBURSED_AMT","CURRENT_ROI","NET_RECEIVABLE","NET_LTV","FORECLOSURE"]
		refined_data = data.copy()
		refined_data = refined_data.sort_values(["AGREEMENTID","CUSTOMERID"],ascending = True)
		refined_data = refined_data.groupby(["AGREEMENTID","CUSTOMERID"],sort=False).last().reset_index()
		refined_data = refined_data.drop("category_id",1)
		for i in refined_data["AGREEMENTID"]:
			j = training_data["AGREEMENTID"][count]
			if i == j:
				idx = refined_data.AGREEMENTID[refined_data.AGREEMENTID == j].index.tolist()
				refined_data["FORECLOSURE"][idx] = training_data["FORECLOSURE"][count]
				count += 1
			else:
				refined_data = refined_data[refined_data.AGREEMENTID != i]
			print(str(i)+", "+str(j)+", "+str(count))
		pickle_out = open("refined.pkl","wb")
		pickle.dump(refined_data,pickle_out)
		pickle_out.close()
	else:
		f1 = open("refined.pkl","rb")
		refined_data = pickle.load(f1)
		f1.close()
	columns = ["AGREEMENTID","CUSTOMERID","LOAN_AMT","NET_DISBURSED_AMT","CURRENT_ROI","NET_RECEIVABLE","NET_LTV","FORECLOSURE"]
	#print(refined_data.dtypes)




	# test data
	if os.path.isfile('./refined_test.pkl') == False:
		count=0
		refined_test_data = data.copy()
		refined_test_data = refined_test_data.sort_values(["AGREEMENTID","CUSTOMERID"],ascending = True)
		refined_test_data = refined_test_data.groupby(["AGREEMENTID","CUSTOMERID"],sort=False).last().reset_index()
		refined_test_data = refined_test_data.drop("category_id",1)
		refined_test_data = refined_test_data.drop("FORECLOSURE",1)
		for i in refined_test_data["AGREEMENTID"]:
			j = test_data["AGREEMENTID"][count]
			if i == j:
				count +=1
			else:
				refined_test_data = refined_test_data[refined_test_data.AGREEMENTID != i]
			print(i)
		pickle_out = open("refined_test.pkl","wb")
		pickle.dump(refined_test_data,pickle_out)
		pickle_out.close()
	else:
		f2 = open("refined_test.pkl","rb")
		refined_test_data = pickle.load(f2)
		f2.close()


	# training and execution of the model
	model = Logistic(refined_data,columns)
	predicted_data = prediction(model,refined_test_data)

	#writing the output file 
	df = refined_test_data.copy()
	df = df["AGREEMENTID"].reset_index(drop=True)
	df = pd.concat([df,predicted_data],1)
	df.to_csv("result.csv",index=False,sep=',',encoding='utf-8')








