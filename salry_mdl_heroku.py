#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 00:50:06 2020

@author: utkarshkushwaha
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir('/Users/utkarshkushwaha/Desktop/Spyderworkspace/DataScience_rvidon')
#creating df
df = pd.read_csv('hiring.csv')

#renaming features
df = df.rename(columns={'test_score(out of 10)':'test_score','interview_score(out of 10)':'interview_score'})
df['experience'].fillna(0,inplace=True)
df['test_score'].fillna(df.test_score.mean(), inplace = True)
df.head()

#dependent variables
X  = df.iloc[:,:3]

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]
X['experience'] = X['experience'].apply(lambda x:convert_to_int(x))
X['test_score'] = X['test_score'].apply(lambda x: np.round_(x))
X.head()

#independent feature
y = df.iloc[:,-1:]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)
#regressor.score(X,y)

#dumping as pickle file
with open('salary_prediction.pkl','wb') as f:
    pickle.dump(regressor,f)
    
model = pickle.load(open('salary_prediction.pkl','rb'))
res = model.predict([[1,6,7]])
s = np.round(res[0],2)
