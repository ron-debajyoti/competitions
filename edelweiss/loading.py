'''
Objective is primarily to arrive at a propensity to foreclose and balance transfer 
an existing loan based on lead indicators such as demographics, internal behavior 
and performance on all credit lines; along with the estimated ‘Time to Foreclose’
'''


import pandas as pd 
import numpy as np 
import pickle

def loading_customerdata():
	#preloading data info of the customers
	customer_data = pd.ExcelFile("LMS_31JAN2019.xlsx")
	df= customer_data.parse("LMS_HLLAP")
	pickle_out = open("customer_data.pkl","wb")
	pickle.dump(df,pickle_out)
	pickle_out.close()

