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
import cache_functions
import functions


######################
## Import Libraries ##
######################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import streamlit as st
import altair as alt
import mplcursors
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import matplotlib.dates as mpl_dates
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Collapse on start
st.set_page_config(initial_sidebar_state="collapsed")


#######################
## Global Variables  ##
#######################

global df_crime
global df_airbnb
global df_airbnb_original
global df_crime_original
global df_descriptions
global user_date


##################
## Load in Data ##
##################
df_airbnb_temp = cache_functions.load_airbnb()
df_airbnb = df_airbnb_temp.copy() # Copy the df

df_airbnb_original_temp = cache_functions.load_airbnb_original()
df_airbnb_original = df_airbnb_original_temp.copy() # Copy the df

df_crime_temp = cache_functions.load_crime()
df_crime = df_crime_temp.copy() # Copy the df

df_crime_orginial_temp = cache_functions.load_crime_original()
df_crime_orginial = df_crime_orginial_temp.copy() # Copy the df

df_descriptions_temp = cache_functions.load_crime_descriptions()
df_descriptions = df_descriptions_temp.copy() # Copy the df


#######################
## Streamlit Sidebar ##
#######################

with st.sidebar.form(key ='Form1'):

    # Add title
    st.title("New York Data")
    st.write('### Set Filters - Global')
    
    # Drop down menus
    borough = st.selectbox("Borough", ('All',
                                                   'Bronx',
                                                   'Manhattan',
                                                   'Queens',
                                                   'Brooklyn',
                                                   'Staten Island'))

    st.write('### Set Filters - Time Series')
    crime_type = st.selectbox("Type of Crime", ('All',
                                                           'Misdemeanor',
                                                           'Felony',
                                                           'Violation'))

    # Time of day input
    time_of_day = st.selectbox("Time of Day", ('All',
                                                           'Morning',
                                                           'Evening',
                                                           'Afternoon',
                                                           'Late Night'))

    # Date input
    min_date = pd.to_datetime(pd.to_datetime(df_crime['CMPLNT_FR_DT']).min())
    max_date = pd.to_datetime(pd.to_datetime(df_crime['CMPLNT_FR_DT']).max() - pd.DateOffset(days=30))
    user_date = st.date_input("Select a Date", value=min_date, min_value=min_date, max_value=max_date)

    # Sliders
    st.write('### Set Filters - Airbnb')

    values=[]
    values = st.slider('Airbnb Price', 
                               float(df_airbnb['price'].min()), 
                               float(df_airbnb['price'].max()),
                               (float(df_airbnb['price'].min()), float(df_airbnb['price'].max())))
    priceLower = values[0] # Assign upper and lower values for filtering
    priceUpper = values[1]

    values = st.slider('Airbnb Review Rate Number',
                               0,
                               int(df_airbnb['review rate number'].max()),
                               (0, int(df_airbnb['review rate number'].max())))
    reviewLower = values[0] # Assign upper and lower values for filtering
    reviewUpper = values[1]

    # Checkboxes
    hostVerified = st.checkbox('Airbnb Host Identity Verified')
    if hostVerified:
        df_airbnb = df_airbnb[df_airbnb['host_identity_verified'] == 1]

    instantBookable = st.checkbox('Airbnb Instant Bookable')
    if instantBookable:
        df_airbnb = df_airbnb[df_airbnb['instant_bookable'] == 1]

    # Apply button
    submitted = st.form_submit_button(label = "Apply Filters")
    

####################################
## Filter DF Based on User Input  ##
####################################

df_airbnb = df_airbnb[df_airbnb['price'] >= priceLower]
df_airbnb = df_airbnb[df_airbnb['price'] <= priceUpper]
df_airbnb = df_airbnb[df_airbnb['review rate number'] >= reviewLower]
df_airbnb = df_airbnb[df_airbnb['review rate number'] <= reviewUpper]
if (borough != 'All'):
    df_airbnb = df_airbnb[df_airbnb['neighbourhood group'] == borough]
    df_crime = df_crime[df_crime['BORO_NM'] == borough.upper()]
if (crime_type != 'All'):
    df_crime = df_crime[df_crime['LAW_CAT_CD'] == borough.upper()]
if (time_of_day != 'All'):
    df_crime = df_crime[df_crime['Time_Category'] == time_of_day]


#####################
## Page Functions  ##
#####################

def welcomePage():
    st.markdown("""<h1 style='text-align: center;'>Welcome to our MSU CSE881 Final Project 😄</h1>""", 
        unsafe_allow_html=True)
    st.markdown("Note: open up the side panel on the left side to explore the different pages.")

def heatMapPage():
    global df_airbnb


    ##################
    ## Display Map  ##
    ##################    

    # Create a base map
    map_center = [df_airbnb['lat'].mean(), df_airbnb['long'].mean()]  # Use the mean coordinates as the initial center
    my_map = folium.Map(location=map_center, zoom_start=12)

    # Add MarkerCluster for Airbnb points
    marker_cluster = MarkerCluster().add_to(my_map)

    for index, row in df_airbnb.iterrows():
        # Create a popup with price information
        popup_text = f"Nightly Cost: ${row['price']} Rating: {row['review rate number']}"
        popup = folium.Popup(popup_text, parse_html=True, max_width=300)

        # Create a marker with the popup and add it to the MarkerCluster
        folium.Marker([row['lat'], row['long']], popup=popup).add_to(marker_cluster)

    # Display the map
    st.markdown("### New York Interactive Airbnb Map")
    st_data = st_folium(my_map, width=725)

def dfOverviewPage():
    global df_crime_orginial
    global df_descriptions
    global df_airbnb_original
    

    ###################
    ## Display DF's  ##
    ###################
    
    # Crime
    st.title('Crime Dataset')
    page_size = 10000
    page_number = st.number_input('Select page number', min_value=1, max_value=len(df_crime_orginial) // page_size + 1, value=1)
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    st.dataframe(df_crime_orginial.iloc[start_idx:end_idx])
    
    # Crime Descriptions
    st.title('Crime Dataset Descriptions')
    st.dataframe(df_descriptions)
    
    # Airbnb
    st.title('Airbnb Housing Dataset')
    page_size2 = 10000
    page_number2 = st.number_input('Select page number', min_value=1, max_value=len(df_airbnb_original) // page_size2 + 1, value=1)
    start_idx2 = (page_number2 - 1) * page_size2
    end_idx2 = start_idx2 + page_size2
    st.dataframe(df_airbnb_original.iloc[start_idx2:end_idx2])

    # Airbnb Descriptions
    st.title('Airbnb Housing Dataset Descriptions')
    original_columns = [
        'id', 'NAME', 'host id', 'host_identity_verified', 'host name',
        'neighbourhood group', 'neighbourhood', 'lat', 'long', 'country',
        'country code', 'instant_bookable', 'cancellation_policy', 'room type',
        'Construction year', 'price', 'service fee', 'minimum nights',
        'number of reviews', 'last review', 'reviews per month',
        'review rate number', 'calculated host listings count',
        'availability 365', 'house_rules', 'license'
    ]
    column_descriptions = {
        'id': 'Listing ID',
        'NAME': 'Name of Listing',
        'host id': 'Host ID',
        'host name': 'Name of Host',
        'neighbourhood group': 'Borough',
        'neighbourhood': 'Area',
        'lat': 'Latitude Coordinates',
        'long': 'Longitude Coordinates',
        'room type': 'Listing Space Type',
        'price': 'Price in Dollars',
        'service fee': 'Service Fee', 
        'minimum nights': 'Minimum Nights',
        'number of reviews': 'Number of Reviews', 
        'last review': 'Last Review Date', 
        'reviews per month': 'Reviews per Month',
        'review rate number': 'Review Rate Number', 
        'calculated host listings count': 'Number of Host Listings',
        'availability 365': 'Whether the booking is availabile 365 days', 
        'house_rules': 'House Rules', 
        'license': 'License'
    }
    df_description = pd.DataFrame({
        'Column': original_columns,
        'Description': [column_descriptions.get(col, col) for col in original_columns]
    })
    st.dataframe(df_description, width=800)

def crimeForecastPage():
    global df_crime
    global user_date

    # Set title of page
    if len(df_crime['BORO_NM'].unique()) > 1:
        num_boroughs = 'All Boroughs'
    else:
        num_boroughs = df_crime['BORO_NM'].unique()[0]


    ##########################
    ## Display Time Series  ##
    ##########################

    # ARIMA
    if user_date is not None:

        # Set title
        st.title(f'ARIMA Time Series Forecasting for Daily Crime Rate ({user_date.year}) ({num_boroughs})')

        # Filter the DataFrame for the selected year
        df_temp = df_crime[df_crime['Year'] == user_date.year].copy()  # Use copy to avoid SettingWithCopyWarning

        # Create a datetime column from the 'Year', 'Month', and 'Day' columns
        df_temp['Date'] = pd.to_datetime(df_temp[['Year', 'Month', 'Day']])

        # Group by date and count the number of occurrences (daily crime rate)
        daily_crime_rate = df_temp.groupby('Date').size()

        # Convert the index to datetime
        daily_crime_rate.index = pd.to_datetime(daily_crime_rate.index)

        try:
            # Standardize the data
            scaler = StandardScaler()
            daily_crime_rate_standardized = scaler.fit_transform(daily_crime_rate.values.reshape(-1, 1))
            daily_crime_rate_standardized = pd.Series(daily_crime_rate_standardized.flatten(), index=daily_crime_rate.index)

            # Fit ARIMA model with explicit frequency
            arima_model = ARIMA(daily_crime_rate_standardized, order=(5, 1, 0), freq='D')
            arima_fit = arima_model.fit()

            # Forecast for the user-specified date and the following week
            user_date = pd.to_datetime(user_date)  # Fully convert to avoid errors
            forecast_start_date = user_date
            forecast_end_date = user_date + pd.DateOffset(days=30)
            arima_forecast = arima_fit.predict(start=forecast_start_date, end=forecast_end_date, typ='levels')

            # Inverse transform the forecasted values to get back to the original scale
            arima_forecast_original_scale = scaler.inverse_transform(arima_forecast.values.reshape(-1, 1))
            arima_forecast_original_scale = pd.Series(arima_forecast_original_scale.flatten(), index=arima_forecast.index)

            # Calculate Mean Squared Error
            mse = mean_squared_error(daily_crime_rate.loc[forecast_start_date:forecast_end_date], arima_forecast_original_scale)

            # Plot original data and forecast
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(daily_crime_rate, label='Original Data', marker='o', linestyle='-', color='b')
            ax.plot(arima_forecast_original_scale, label='ARIMA Forecast', marker='o', linestyle='--', color='r')
            ax.scatter([user_date], [arima_forecast_original_scale.loc[user_date]], color='g', label='User Date Prediction')
            ax.set_title(f'ARIMA Forecasting for Daily Crime Rate ({user_date.year})')
            ax.set_xlabel('Date')
            ax.set_ylabel('Crime Rate')

            # Format x-axis labels
            date_format = mpl_dates.DateFormatter('%Y-%m-%d')  # Adjust the format as needed
            ax.xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability

            ax.legend()
            ax.grid(True)

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Print the predicted crime rates for the user-specified date and the following week
            st.write(f"Predicted Crime Rate for {user_date.strftime('%Y-%m-%d')}: {arima_forecast_original_scale.loc[user_date]:.2f}")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")

            # Display forecast table
            st.write("\nPredicted Crime Rates for the Following 30 Days:")
            arima_forecast_dates = pd.to_datetime(arima_forecast.index)
            forecast_table = pd.DataFrame({
                'Date': arima_forecast_dates.strftime('%Y-%m-%d'),
                'Predicted Average Crime Rate': arima_forecast_original_scale.values,
                'Actual Average Crime Rate': daily_crime_rate.loc[forecast_start_date:forecast_end_date],
                'Difference': abs(arima_forecast_original_scale.values - daily_crime_rate.loc[forecast_start_date:forecast_end_date])})
            st.dataframe(forecast_table.set_index('Date'), width=800)

        except Exception as e:
            # Handle the exception
            st.error(f"An error occurred: {str(e)}")
            st.error("Not enough data: please refilter the data")

    else:
        st.write("Note: Select a date to explore the by-year ARIMA model portion of this project.")

    # SARIMA
    if True:

        # Set title
        st.title(f'SARIMA Time Series Forecasting for Daily Crime Rate (2014-2016) (All Boroughs)')

        # Create a copy of the dataframe
        df_temp = df_crime.copy()

        # Remove rows where 'Year' is less than 2014
        df_temp = df_temp[df_temp['Year'] >= 2014]

        # Create a datetime column from the 'Year', 'Month', and 'Day' columns
        df_temp['Date'] = pd.to_datetime(df_temp[['Year', 'Month', 'Day']])

        # Group by date and count the number of occurrences (daily crime rate)
        daily_crime_rate = df_temp.groupby('Date').size()

        # Convert the index to datetime
        daily_crime_rate.index = pd.to_datetime(daily_crime_rate.index)

        # Define the training period
        training_end = pd.to_datetime('2016-8-30')

        # Number of periods (days) to forecast
        forecast_periods = 30  

        # Split the data into training and test sets
        train = daily_crime_rate[:training_end]
        test = daily_crime_rate[training_end + pd.Timedelta(days=1):training_end + pd.Timedelta(days=forecast_periods)]

        # Standardize the data
        scaler = StandardScaler()
        train_std = scaler.fit_transform(train.values.reshape(-1, 1))
        test_std = scaler.transform(test.values.reshape(-1, 1))

        # Store the actual test data for evaluation
        # evaluation_results[selected_borough] = test

        if not train.empty:
            
            # Fit a SARIMA model on the training set
            try:
                sarima_model = SARIMAX(train_std, 
                    order=(1, 1, 2), 
                    seasonal_order=(0, 2, 2, 32))
                fitted_model = sarima_model.fit()

                # Forecast the next month
                forecast_std = fitted_model.get_forecast(steps=forecast_periods).predicted_mean
                evaluation_result_std = test_std

                # Inverse transform the standardized forecast to get the actual values
                forecast = scaler.inverse_transform(forecast_std.reshape(-1, 1)).flatten()
                evaluation_result = scaler.inverse_transform(evaluation_result_std.reshape(-1, 1)).flatten()

                # Plot the training data, the forecast, and the actual observations
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train.index, train, label=f'Crime Count (Training)')
                ax.plot(test.index, test, label=f'Crime Count (Actual)', color='orange')
                ax.plot(test.index, forecast, label=f'Forecast', color='green')
                ax.set_title(f'SARIMA Forecasting for Daily Crime Rate')
                ax.set_xlabel('Date')
                ax.set_ylabel('Crime Count')

                # Display the plot in Streamlit
                st.pyplot(fig)

                # Example of evaluating the forecast for the selected borough
                min_length = min(len(forecast), len(evaluation_result))
                if min_length > 0:
                    actual = evaluation_result[:min_length]
                    predicted = forecast[:min_length]
                    mse = mean_squared_error(actual, predicted)
                    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
                    sarima_forecast_dates = pd.to_datetime(test.index)
                    forecast_table = pd.DataFrame({
                        'Date': sarima_forecast_dates.strftime('%Y-%m-%d'), 
                        'Predicted Average Crime Rate': predicted, 
                        'Actual Average Crime Rate': actual,
                        'Difference': abs(predicted - actual)})
                    st.dataframe(forecast_table.set_index('Date'), width=800)

                else:
                    st.warning(f"Not enough data to calculate MSE")

            except Exception as e:
                # Handle the exception
                st.error(f"An error occurred: {str(e)}")
                st.error("Not enough data: please refilter the data")

    return


#####################
## Page Functions  ##
#####################

page_names_to_funcs = {
    "Welcome": welcomePage,
    "Crime Forcasting": crimeForecastPage,
    "Interactive Airbnb Map": heatMapPage,
    "Dataframe Overviews": dfOverviewPage
}
st.sidebar.write('### Page Selection')
selected_page = st.sidebar.selectbox("Select a Page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


#########
## CSS ##
#########

padding = 2
st.markdown(f""" <style>
    .st-emotion-cache-16txtl3 {{
        padding-top: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)  


