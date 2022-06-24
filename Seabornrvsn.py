# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Tue May 26 21:16:25 2020

@author: utkarshkushwaha
"""
import os 
import pandas as pd
os.chdir('/Users/utkarshkushwaha/Desktop/Spyderworkspace/DataScience_rvidon')
import seaborn

df = sns.load_dataset('https://www.kaggle.com/ranjeetjain3/seaborn-tips-dataset')
df = pd.read_csv('tips.csv')

#correlation
df.corr()

#heatmap of correlatiion between numeric features
sns.heatmap(df.corr())

#jionplot
#basically it is three dimensional plotting
#this is used for two correlated variable analyis 

sns.jointplot(x = df['total_bill'], y = df['tip'], data = df, kind = 'hex' )
sns.jointplot(x ='total_bill', y = 'tip', data = df, kind= 'hex' )
sns.jointplot(x ='total_bill', y = 'tip', data = df, kind= 'reg' )

#pairplot
#this is also used for bivariat analysis


sns.pairplot(df)
#gives all plotting  posible combination with all feature

sns.pairplot(df ,hue = 'sex')#picking particular feature
sns.pairplot(df,hue='smoker')
df['smoker'].value_counts()


#distplot 
#this is used to draw hstogram 

sns.distplot(df['tip'])#this giv epercentage of people giving tip
sns.distplot(df['tip'], kde= False, bins =10)

df.info()
df.describe()




#Categorical plotting in SNS


#countplot
#this gives simple count of particular categorical feature
sns.countplot('sex',data = df)
sns.countplot('day', data = df)

#barplot
#this gives count on the basis of different axis

sns.barplot(x='sex', y = 'total_bill', data=df)


#Boxplot(gives info about percentile of data)
sns.boxplot(x='day',y ='total_bill', data =df, palette='rainbow', orient = 'v')


#voilin plot
#this help us to see data distribution in KDE
sns.violinplot(x ='total_bill', y= 'day', data= df)

from sklearn import datasets

iris = datasets.load_iris()
df = sns.load_dataset('iris')

df.info()
df.head()

#pairplot
#jointplot
#distplot
sns.pairplot(df)

df1= pd.read_csv('Bengaluru_House_Data.csv')






































