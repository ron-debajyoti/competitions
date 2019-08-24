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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics,preprocessing

scaler = preprocessing.MaxAbsScaler()
pd.set_option('display.max_columns', None)



def Logistic(training_data,columns):
	x_train,x_test,y_train,y_test = train_test_split(training_data[columns[:-1]],training_data[columns[-1]],random_state=0,train_size = 0.7)
	x_train = preprocessing.scale(x_train)
	x_train = scaler.fit_transform(x_train)

	x_test = preprocessing.scale(x_test)
	x_test = scaler.fit_transform(x_test)

	multi_logr = LogisticRegressionCV(cv=5,fit_intercept = True,penalty = "l2",solver="lbfgs").fit(x_train,y_train)
	print(metrics.accuracy_score(y_test,multi_logr.predict(x_test)))
	print("\n")
	return multi_logr


def SVM_Grid(training_data,columns):
	x_train,x_test,y_train,y_test = train_test_split(training_data[columns[:-1]],training_data[columns[-1]],random_state=0,train_size = 0.7)
	x_train = preprocessing.scale(x_train)
	x_train = scaler.fit_transform(x_train)

	x_test = preprocessing.scale(x_test)
	x_test = scaler.fit_transform(x_test)

	svc_model = SVC(probability = True,tol = 1e-5,kernel = "linear")
	Cs = [0.001, 0.01, 0.1, 1, 10]
	gammas = [0.001, 0.01, 0.1, 1]
	param_grid = {'C': Cs, 'gamma' : gammas}
	print("1*-")
	grid_search = GridSearchCV(svc_model,param_grid,cv = 5)
	print("2*-")
	grid_search = grid_search.fit(x_train,y_train)

	print(metrics.accuracy_score(y_test,grid_search.predict(x_test)))
	print("\n")
	return grid_search



def Forest(training_data,columns):
	x_train,x_test,y_train,y_test = train_test_split(training_data[columns[:-1]],training_data[columns[-1]],random_state=0,train_size = 0.7)
	x_train = preprocessing.scale(x_train)
	x_train = scaler.fit_transform(x_train)

	x_test = preprocessing.scale(x_test)
	x_test = scaler.fit_transform(x_test)

	parameters = {'bootstrap': False,
              	'min_samples_leaf': 10,
              	'n_estimators': 50,
              	'min_samples_split': 30,
              	'max_features': 'sqrt',
              	'max_depth': 20}
	model = RandomForestClassifier(**parameters)
	model.fit(x_train,y_train)
	print(metrics.accuracy_score(y_test,model.predict(x_test)))
	return model




def prediction(model_name,test_data):
	predicted_data = model_name.predict_proba(scaler.fit_transform(preprocessing.scale(test_data)))[:,1]
	l = predicted_data.tolist()
	predicted_data = np.reshape(predicted_data,(len(l),-1))
	pr_data = pd.DataFrame({"FORECLOSURE":np.array(predicted_data).flatten()})
	return pr_data








if __name__ == "__main__":
	#preloading data info of the customers
	#loading.loading_customerdata()
	#loading.customer_salary_data()
	with open("customer_data.pkl","rb") as f:
		data = pickle.load(f)
		columns = ["AGREEMENTID","CUSTOMERID","LOAN_AMT","NET_DISBURSED_AMT","CURRENT_ROI","NET_RECEIVABLE","BALANCE_EXCESS","BALANCE_TENURE","NET_LTV","FOIR"]
		data = data[columns]
		for column in data.columns:
			data = data[pd.notnull(data[column])]
		data['category_id'] = data['AGREEMENTID'].factorize()[0]
		data["AGREEMENTID"][0] = 11220001
		data["FORECLOSURE"] = 0.0
		data["NETTAKEHOMEINCOME"] =0.0

	with open("customer.pkl","rb") as cus_f:
		custo = pickle.load(cus_f)
		cols = ["CUSTOMERID","NETTAKEHOMEINCOME"]
		for col in custo.columns:
			custo = custo[pd.notnull(custo[col])]

	# reading training_data and test_data from the respective .csv files
	training_data = pd.read_csv("train_foreclosure.csv",sep = ",")
	training_data = training_data[pd.notnull(data["AGREEMENTID"])]
	test_data = pd.read_csv("test_foreclosure.csv",sep = ",")
	test_data = test_data[pd.notnull(test_data["AGREEMENTID"])]
	
	

##############################################################################################################################################################
	#refined_data
	if os.path.isfile('./refined.pkl') == False :
		count = 0 # refers to the count of the index of the cell from training_data 
		columns = ["AGREEMENTID","CUSTOMERID","NETTAKEHOMEINCOME","LOAN_AMT","NET_DISBURSED_AMT","CURRENT_ROI","NET_RECEIVABLE","BALANCE_EXCESS","BALANCE_TENURE","NET_LTV","FOIR","FORECLOSURE"]
		refined_data = data.copy()
		refined_data = refined_data.sort_values(["AGREEMENTID","CUSTOMERID"],ascending = True)
		refined_data = refined_data.groupby(["AGREEMENTID","CUSTOMERID"],sort=False).last().reset_index()
		refined_data = refined_data.drop("category_id",1)
		for i in refined_data["AGREEMENTID"]:
			j = training_data["AGREEMENTID"][count]
			# refined_data["CUSTOMERID"][count] returns the customer id 
			if i == j:
				idx = refined_data.AGREEMENTID[refined_data.AGREEMENTID == j].index.tolist()
				refined_data["FORECLOSURE"][idx] = training_data["FORECLOSURE"][count]
				count += 1
			else:
				refined_data = refined_data[refined_data.AGREEMENTID != i]
			print(str(i)+", "+str(j)+", "+str(count))

		refined_data = refined_data.reset_index()
		pickle_out = open("refined.pkl","wb")
		pickle.dump(refined_data,pickle_out)
		pickle_out.close()
	else:
		f1 = open("refined.pkl","rb")
		refined_data = pickle.load(f1)
		f1.close()
	columns = ["AGREEMENTID","CUSTOMERID","NETTAKEHOMEINCOME","LOAN_AMT","NET_DISBURSED_AMT","CURRENT_ROI","NET_RECEIVABLE","BALANCE_EXCESS","BALANCE_TENURE","NET_LTV","FOIR","FORECLOSURE"]



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

		refined_test_data = refined_test_data.reset_index() 
		pickle_out = open("refined_test.pkl","wb")
		pickle.dump(refined_test_data,pickle_out)
		pickle_out.close()
	else:
		f2 = open("refined_test.pkl","rb")
		refined_test_data = pickle.load(f2)
		f2.close()



	#print(refined_test_data.count)
	#print("For training data : \n")
	#print(refined_data.count)

############################################################################################################################################################################
	# adding the customer salaries wrt the transactions 
	# off.txt represents if customer salaries have been added or not 

	'''
	if os.path.isfile('./refined.pkl') == True:
		if os.path.isfile("./off.txt") == False:
			refined_data = refined_data.drop("index",1)
			refined_data = refined_data.reset_index(drop = True)
			for i,row in refined_data.iterrows():
				try:
					idx2 = custo.CUSTOMERID[custo.CUSTOMERID == row["CUSTOMERID"]].index.tolist()
					refined_data["NETTAKEHOMEINCOME"][i] = custo["NETTAKEHOMEINCOME"][idx2]
					print(str(refined_data["CUSTOMERID"][i])+", "+ str(refined_data["NETTAKEHOMEINCOME"][i]) +",* "+str(i))
				except ValueError:
					refined_data["NETTAKEHOMEINCOME"][i] = -1.0
			#refined_data = refined_data[refined_data["NETTAKEHOMEINCOME"] >=0.0 ]
			#os.remove("refined.pkl")
			pickle_out = open("refined.pkl","wb")
			pickle.dump(refined_data,pickle_out)
			pickle_out.close()
		else:
			f1 = open("refined.pkl","rb")
			refined_data = pickle.load(f1)
			f1.close()
	else:
		print("File not found. Restart the compilation ")

	#print(refined_data.head(20))

	
	if os.path.isfile('./refined_test.pkl') == True:
		if os.path.isfile("./off2.txt") == False:
			refined_test_data = refined_test_data.drop("index",1)
			refined_test_data = refined_test_data.reset_index(drop = True)
			for i,row in refined_test_data.iterrows():
				try:
					idx2 = custo.CUSTOMERID[custo.CUSTOMERID == row["CUSTOMERID"]].index.tolist()
					refined_test_data["NETTAKEHOMEINCOME"][i] = custo["NETTAKEHOMEINCOME"][idx2]
					print(str(refined_test_data["CUSTOMERID"][i])+", "+ str(refined_test_data["NETTAKEHOMEINCOME"][i]) +",* "+str(i))
				except ValueError:
					refined_test_data["NETTAKEHOMEINCOME"][i] = -1.0
			#os.remove("refined_test.pkl")
			pickle_out = open("refined_test.pkl","wb")
			pickle.dump(refined_test_data,pickle_out)
			pickle_out.close()
			#file1 = open("off2.txt","w")
			#file1.write("1")
			#file1.close()
		#refined_test_data = refined_test_data[refined_data["NETTAKEHOMEINCOME"] >=0.0 ]
		else:
			f2 = open("refined_test.pkl","rb")
			refined_data = pickle.load(f2)
			f2.close()
	else:
		print("File not found. Restart the compilation ")


#print(refined_test_data.head(20))


	print(refined_test_data.head(20))
	print(refined_data.head(20))
	'''



	# training and execution of the model
	#model = Logistic(refined_data,columns)
	#model = SVM_Grid(refined_data,columns)
	model = Forest(refined_data,columns)
	predicted_data = prediction(model,refined_test_data)

	#writing the output file 
	df = refined_test_data.copy()
	df = df["AGREEMENTID"].reset_index(drop=True)
	df = pd.concat([df,predicted_data],1)
	df.to_csv("result.csv",index=False,sep=',',encoding='utf-8')
	
