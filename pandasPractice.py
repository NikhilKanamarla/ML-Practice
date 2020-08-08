import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import tensorflow as tf
import keras
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import re



if __name__ == '__main__':
    #store data in CSV 
    dataFrame = pd.read_csv('pokemon_data.csv')
    #allows you to specify which rows to print
    print(dataFrame.head(5))
    print(dataFrame.tail(5))
    #print headers
    store = dataFrame.columns
    store = len(store)
    print(store)
    print(dataFrame.columns)
    #read a specific column
    print(dataFrame['Name'])
    #get multiple columns
    print(dataFrame[['Name','HP']])
    #get a specific row
    print()
    print("printing the first row")
    print(dataFrame.iloc[1])
    #get a specific row and column
    print()
    print("get first row and first column") 
    print(dataFrame.iloc[0,1])
    print("get a specific section of a row or column")
    print(dataFrame.iloc[0,1:6])
    #get specific rows from a certain column 
    print()
    print("get specific rows of a column")
    print(dataFrame.loc[dataFrame['Type 1'] == "Fire"])
    #get statistics on the data 
    print()
    print("standard statistics on data ", dataFrame.describe())
    #sort entries by ordering a specific column 
    print("\n values in CSV can also be sorted")
    print(dataFrame.sort_values('Name',ascending=True))
    print("sort values in CSV based on multiple factors")
    print(dataFrame.sort_values(['HP','Attack','Defense'],ascending=False))
    #modifying data and adding a new column 
    print("\n adding a new column")
    dataFrame['Total'] = dataFrame['HP'] + dataFrame['Attack'] + dataFrame['Defense']
    print(dataFrame.head(5))
    #removing a column
    print('\n removing a column') 
    dataFrame = dataFrame.drop(columns=['Total'])
    print(dataFrame.head(5))
    #reordering the columns (include a method for csv generalizability 
    print("\n reorder the columns to your convience")
    dataFrame['Total'] = dataFrame['HP'] + dataFrame['Attack'] + dataFrame['Defense']
    columns = list(dataFrame.columns)
    dataFrame = dataFrame[columns[0:4] + [columns[-1]] + columns[4:12]]
    print(dataFrame.head(5))
    #export to CSV (include in method to allow people to test their reordered csv)
    dataFrame.to_csv("pokemon_data2.csv",index=False,sep=",")
    #renaming the columns (include in method )
    print("\n rename the columns")
    #TBD
    
    #filtering data in pandas
    print("\n get rows from a column that match certain criteria")
    df = dataFrame.loc[(dataFrame['Type 1'] == "Grass") & (dataFrame['Type 2'] == "Poison")]
    #clean up index counts 
    df = df.reset_index()
    print(df)
    #filtering by a more specific parameter using contains  ~ gives opposite
    df2 = dataFrame.loc[~dataFrame['Name'].str.contains('Mega')]
    print("\n specific filtering")
    print(df2)
    #filtering while ignoring case using regular expressions
    df3 = dataFrame.loc[dataFrame['Type 1'].str.contains('fire | grass',flags=re.I,regex=True)]
    print(df3)
    #filtering based on the start of a row 
    #TBD getting specific folders of cowc (ex Utah and DARPA)
    print("filtering based on the start of a row")
    df4 = dataFrame.loc[dataFrame['Name'].str.contains('^pi[a-z]*',flags=re.I,regex=True)]
    print(df4)

    #conditional changes
    print("\n change the name of specific parts of a row") 
    dataFrame3 = dataFrame.copy()
    dataFrame3.loc[dataFrame3['Type 1'] == 'Fire','Type 1'] = 'Flamer'
    print(dataFrame3.head())

    #Aggrerate stats
    print("\n use to get stats on a specific group of data")
    dataFrame4 = pd.read_csv("pokemon_data.csv")
    #mean,sum,count can work 
    print(dataFrame4.groupby(['Type 1']).mean().sort_values('Defense'))
    #multiple parameters
    print(dataFrame4.groupby(['Type 1','Type 2']).count())

    #Working with large datasets 
    print("loading in pieces")
    for store in pd.read_csv('pokemon_data.csv',chunksize=100):
        print("Chunk Dataframe")
        print(store)



    


