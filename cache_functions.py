#######################
##                   ##
##  Nickolaus White  ##
##  Stephen Lee      ##
##  Mark Carravallah ##
##      CSE881       ##
##                   ##
#######################


######################
## Import Files ##
######################
import functions


######################
## Import Libraries ##
######################
import pandas as pd
from datetime import datetime
import streamlit as st
import geopandas as gpd
import os 

################################
## Clean & Prep Data - Airbnb ##
################################

@st.cache_data
def load_airbnb():

    # Load in data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = dir_path + r'\Data\Airbnb_Open_Data.csv'
    df = pd.read_csv(data_path, low_memory=False)
    
    # Specify the columns of interest
    columns_of_interest = ['neighbourhood group', 'lat', 'long', 'price', 'host_identity_verified', 'instant_bookable', 'review rate number']

    # Drop rows with NaN values in specified columns
    df = df.dropna(subset=columns_of_interest)
    df = df[columns_of_interest]

    # Fix row in neighbourhood group that has 'brookln' instead of 'brooklyn'
    df['neighbourhood group'] = df['neighbourhood group'].replace('brookln', 'Brooklyn')
    df['neighbourhood group'].value_counts()

    # Fix the pricing format issues
    df['price'] = df['price'].apply(functions.fix_price)
    df['price'] = df['price'].astype(float)

    # Convert `host_identity_verified` to categorical column then convert to binomial (0 & 1)
    df['host_identity_verified'] = df['host_identity_verified'].astype('category')
    df['host_identity_verified'] = df['host_identity_verified'].cat.codes

    # Convert `instant_bookable` to 0 and 1 categorical column
    df['instant_bookable'] = df['instant_bookable'].astype('category')
    df['instant_bookable'] = df['instant_bookable'].cat.codes

    # Drop columns where value counts is equal to 1
    for col in df.columns:
        if df[col].value_counts().shape[0] == 1:
            df = df.drop(col, axis=1)
            
    return df

@st.cache_data
def load_airbnb_original():

    # Load in data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = dir_path + r'\Data\Airbnb_Open_Data.csv'
    df = pd.read_csv(data_path, low_memory=False)
    
    return df

###############################
## Clean & Prep Data - Crime ##
###############################

@st.cache_data
def load_crime():
    
    # Load in data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    data_path = dir_path + r'\Data\NYPD_Complaint_Data_Historic.csv'
    df1 = pd.read_csv(data_path, low_memory=False)
    
    data_path = dir_path + r'\Data\NYPD_Complaint.csv'
    df2 = pd.read_csv(data_path, low_memory=False)

    # Concatenate the 2 DataFrames vertically
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Grab only needed columns
    df = combined_df[['CMPLNT_FR_DT', 
                     'CMPLNT_FR_TM', 
                     'LAW_CAT_CD', 
                     'BORO_NM',
                     'Latitude',
                     'Longitude']]

    # Remove rows with NaN values
    df = df.dropna(subset=['CMPLNT_FR_DT'])

    # Create a copy of the DataFrame to void warnings
    df = df.copy()

    # Apply the categorize_time function to create the 'Time_Category' column
    df['Time_Category'] = df['CMPLNT_FR_TM'].apply(functions.categorize_time)

    # Separate 'CMPLNT_FR_DT' into 'Month', 'Day', and 'Year' columns
    df[['Month', 'Day', 'Year']] = df['CMPLNT_FR_DT'].str.split('/', expand=True)

    # Convert the columns to numeric if needed
    df[['Month', 'Day', 'Year']] = df[['Month', 'Day', 'Year']].apply(pd.to_numeric)

    # Remove rows where 'Year' is less than 2013
    df = df[df['Year'] >= 2013]
    
    return df

@st.cache_data
def load_crime_original():
    
    # Load in data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    data_path = dir_path + r'\Data\NYPD_Complaint_Data_Historic.csv'
    df1 = pd.read_csv(data_path, low_memory=False)
    
    data_path = dir_path + r'\Data\NYPD_Complaint.csv'
    df2 = pd.read_csv(data_path, low_memory=False)
    
    # Concatenate the 2 DataFrames vertically
    df = pd.concat([df1, df2], ignore_index=True)

    return df

@st.cache_data
def load_crime_descriptions():

    # Load in data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = dir_path + r'\Data\Crime_Column_Description.csv'
    df = pd.read_csv(data_path, low_memory=False)

    return df


