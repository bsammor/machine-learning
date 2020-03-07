#-------------------------------------------------------------------------------
#	This code is to preprocess the dataset from the below link:
#	https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data
#	
#	Before running this code, keep the downloaded dataset in a zip file
#	in the same directory as this code. Name the zip file Data.zip	
#
#	Input: (look for this in global variables) 'path' variable
#	Output: [X, Y] values which are preprocessed for training
#
#	Date: 07 Mar 2020

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#	IMPORTS

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob
import os

#-------------------------------------------------------------------------------
#	GLOBAL VARIABLES
#path to the unziped dataset directory 

path = '/home/saarika/Desktop/ML/PRSA_Data_20130301-20170228/'

try: 
	print("\treading from path: ", path, "\t")
	all_files = glob.glob(os.path.join(path, "*.csv"))
except: print("\nNo path specified. Please specify path in preprocess.py\n")

#-------------------------------------------------------------------------------
#	FUNCTIONS

# Function to intialize an empty dictionary to a dictionary with the 
# skeleton of the dataframe
# takes in a dataframe 
# returns a dictionary with the skeleton of the dataframe

def initialize_dict(df):
  d = {}
  for i in range(len(df.columns)):
    d[df.columns[i]] = []
  return d


# Function to create 1 row for the particular date
# input: dictionary(with the skeleton of the df/can have some values ), the dataframe, the date
# returns a dictionary appended with that date's mean data

def MeanofThatDay(init_dict, df3000, date): 
  for x in range(len(df3000.columns)):
    if np.dtype(df3000[df3000.columns[x]]) != np.dtype('O'): #Checking if the column is a String(type : 'O' - meaning 'Object' ) columns 
      init_dict[df3000.columns[x]].append(df3000.loc[df3000.Date == date][df3000.columns[x]].mean())
    else:
      init_dict[df3000.columns[x]].append(df3000.loc[df3000.Date == date][df3000.columns[x]].tolist()[0])
  return init_dict


#Function to Group Wind Direction into 4 groups: NW, NE, SW, SE
#Input: DataFrame with Wind Direction
#Output: DataFrame with grouped wind dir

def GroupWindDir(df):
  df.replace({
      'wd': {
          'N': 'NW', 'WNW': 'NW', 'NNW': 'NW', 'W': 'NW' #For group NW
          'NNE': 'NE', 'ENE' : 'NE', #For group NE
          'E': 'SE', 'ESE': 'SE', 'SSE': 'SE', 'S': 'SE', #For group SE
          'SSW': 'SW', 'WSW': 'SW' #For group SW
      }
  }, inplace=True)

# Function to Group Wind Direction into 4 numerical groups: 1, 2, 3, 4
# Input: DataFrame with Wind Direction
# Output: DataFrame with numerically grouped wind dir
def GroupWindDir_Numbers(df):
  df.replace({
      'wd': {
          'NW': 1, # For group NW
          'NE' : 2, # For group NE
          'SE': 3, # For group SE
          'SW': 4 # For group SW
      }
  }, inplace=True)

# Function to Group Stations into 12 numerical groups: 1, 2,.., 12
# Input: DataFrame with station
# Output: DataFrame with numerically named station
def GroupStation_Numbers(df):
  df.replace({
      'station': {
          'Huairou': 1, 'Aotizhongxin': 2, 'Wanliu': 3, 'Tiantan': 4, 'Changping': 5,
       'Gucheng': 6, 'Dongsi': 7, 'Wanshouxigong': 8, 'Guanyuan': 9, 'Nongzhanguan': 10,
       'Dingling': 11, 'Shunyi': 12
      }
  }, inplace=True)


# Function to perform mean normalization and feature scaling 
# takes in a dataframe.Series 
# returns a df.Series scaled and mean normalized
def scale(column):
  difference =  column - column.mean()
  return difference / column.std()



#-------------------------------------------------------------------------------
#	MAIN

def main():

	print("\nSTARTING PREPROCESSING\n")

	
	#Run this for all stations
	df_from_each_file = (pd.read_csv(f) for f in all_files)
	df = pd.concat(df_from_each_file, ignore_index=True)


	# For station values one by one
	# Change index of 0 for all_files below 
		
	# f = all_files[0]
	# df = pd.read_csv(f)	

	print("\tdropping NaN values of dataframe..\n")
	df.dropna(inplace=True)
	print("\tresetting index of dataframe..\n")
	df.reset_index(inplace=True)

	print("\tcreating 'Date' in dataframe..\n")
	df['Date'] = df.year.astype(str) + '/' + df.month.astype(str) + '/' + df.day.astype(str)
#	df.drop(columns = ['year', 'month', 'day', 'hour', 'No'], inplace=True)
#	df.drop(columns = ['SO2', 'NO2', 'CO', 'O3', 'PM10'], inplace=True)
	df.drop('year', axis = 1, inplace = True)
	df.drop('month', axis = 1, inplace = True)
	df.drop('day', axis = 1, inplace = True)
	df.drop('hour', axis = 1, inplace = True)
	df.drop('No', axis = 1, inplace = True)
	df.drop('SO2', axis = 1, inplace = True)
	df.drop('NO2', axis = 1, inplace = True)
	df.drop('CO', axis = 1, inplace = True)
	df.drop('O3', axis = 1, inplace = True)
	df.drop('PM10', axis = 1, inplace = True)

	print("\ttaking daily averages..\t (This will take some time to run for each station)")

	# Final dataframe that with desired data
	df_final = pd.DataFrame(initialize_dict(df)) #just creating an empty one


	for station in df.station.unique(): # 'station' variable now holds the 12 values one by one
	  dfstation = df.loc[df.station == station] # dfstation is the new dataframe of that particular station
	  # Mean of the day
	  init_dict = initialize_dict(dfstation)
	  for date in dfstation.Date.unique():
	    init_dict = MeanofThatDay(init_dict, dfstation, date)
	  df_final = df_final.append(pd.DataFrame(init_dict), ignore_index=True)
	  print('done with ', station) 

	print("\n\tgiving nos to wind direction..\n")
	# Grouping of wind directions
	GroupWindDir(df_final)
	GroupWindDir_Numbers(df_final)
	print("\tgiving nos to stations..\n")
	# Grouping of stations 
	GroupStation_Numbers(df_final)

	print("\tSending unscaled data into a csv file for future reference..\n")
	# Writing into a csv file for reference
	df_final.to_csv('preproc.csv', index=False)

	print("\tScaling data..\n")


	# Y values from our final set
	Y = np.array(df_final['PM2.5'])
	# Extract the wind direction and stations
	X_wd = np.array([df_final.wd])
	X_station = np.array([df_final.station])
	#dropping Y and X_wd from the DataFrame to be scaled
	#df_final.drop(columns=['PM2.5', 'wd', 'Date', 'station'], inplace=True)
	df.drop('PM2.5', axis = 1, inplace = True)
	df.drop('wd', axis = 1, inplace = True)
	df.drop('Date', axis = 1, inplace = True)
	df.drop('station', axis = 1, inplace = True)

	# Scale dataframe
	for col in df_final.columns:
	  df_final[col] = scale(df_final[col])
	X_temp = np.array(df_final)

	# Regroup the X_wd and X_station with X_temp to get the final X set of features
	X = np.append(X_temp, np.transpose(X_wd), axis=1)
	X = np.append(X, np.transpose(X_station), axis=1)

	print("\nPREPROCESSING FINISHED SUCCESSFULLY\n")



	return [X, Y]



if __name__ == '__main__':
	main()





