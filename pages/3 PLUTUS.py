#creating a page for any stock price prediction

#importing all the requires libraries
import streamlit as st
import pandas as pd 
import numpy as np
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import os
from PIL import Image
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import math # Mathematical functions 
from datetime import date, timedelta, datetime # Date Functions
from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calender dates
import matplotlib.dates as mdates # Formatting dates
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
from tensorflow.keras import Sequential # Deep learning library, used for neural networks
from tensorflow.keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from tensorflow.keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
import seaborn as sns # Visualization
from sklearn.metrics import r2_score
import threading


ico = Image.open("Graphics/Page Related Themes Etc/GT_Logo.png")
st.set_page_config(page_icon=ico,page_title='PLUTUS',layout='wide')

# Define CSS for image alignment
sidebar_style = """
<style>
    .st-emotion-cache-1rtdyuf {
        color: rgb(225, 225, 225);
        align-items: center;
    }
    
    .st-emotion-cache-1egp75f{
        color: rgb(225, 225, 225);
    }

    .element-container st-emotion-cache-lark4g e1f1d6gn4{
        align-items: center;
    }

</style>
"""

# Apply the CSS
st.markdown(sidebar_style, unsafe_allow_html=True)


# Define custom CSS
custom_css = """
<style>
    .custom-selectbox {
        background-color: rgba(0,0,0,.75);
        padding: 5px;
        border-radius: 5px;
    }
    .custom-selectbox p {
        color: white;
        margin: 0; /* Remove default margin */
    }

</style>
"""

# Apply custom CSS
st.markdown(custom_css, unsafe_allow_html=True)


# Define CSS for image alignment
sidebar_style = """
<style>
    .st-emotion-cache-1rtdyuf {
        color: rgb(225, 225, 225);
        align-items: center;
    }
    
    .st-emotion-cache-1egp75f{
        color: rgb(225, 225, 225);
    }

    .element-container st-emotion-cache-lark4g e1f1d6gn4{
        align-items: center;
    }

</style>
"""
# Apply the CSS
st.markdown(sidebar_style, unsafe_allow_html=True)


#creatin a function to convert an image into base64
@st.cache_data
def get_image_as_base64(file):
    with open(file,"rb") as f:
        data=f.read()
    return base64.b64encode(data).decode()

#getting the sidebar image
img = get_image_as_base64("Graphics/Page Related Themes Etc/Sidebar BG Edited.png")

####################################### Managing the background attributes #######################################

# Define the CSS style with the GIF background
css = f'''<style> 

    [data-testid="stHeader"]{{
    background-color: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"]{{
    right: 2rem;
    }}

    [data-testid="stSidebar"]{{
    background-image:url("data:image/png;base64,{img}");
    opacity:1;
    background-size:cover;
    }}

    [data-testid="stSidebarNavLink"]{{
    background-color:rgba(0,0,0,.75);
    background-size:cover;
    }}

    [data-testid="stPageLink-NavLink"]{{
    background-color:rgba(0,131,143,1);
    background-size:cover;
    }}
    
    [data-testid="stSidebarNavSeparator"] {{
    display: none !important;
    }}
    
    </style>'''


# Use st.markdown() to inject the CSS for background and sidebar styling
st.write(css, unsafe_allow_html=True)




########## Create a comprehensive list of tickers that allows users to navigate between the various stocks ##########
#reading the required dataframe
all_data = pd.read_csv("Data/All_Data_Industries.csv")
all_data.set_index('Date',inplace=True)
all_data.index = pd.to_datetime(all_data.index)
all_data.round(4)

##########################Creating filters for the sidebar##########################

sectoral_dict = {'niftyautolist' : 'Nifty AUTO' , 'niftybanklist' : 'Nifty BANK', 'niftyconsumerdurableslist':'Nifty CONSUMER DURABLES',
    'niftyfinancelist':'Nifty FINANCIAL SERVICES', 'niftyfmcglist' :'Nifty FMCG', 'niftyhealthcarelist':'Nifty HEALTHCARE',
    'niftyitlist':'Nifty IT', 'niftymedialist': 'Nifty MEDIA', 'niftymetallist': 'Nifty METAL',
    'niftyoilgaslist':'Nifty OIL & GAS', 'niftypharmalist':'Nifty PHARMA', 'niftyrealtylist':'Nifty REALTY'}

all_data.replace(sectoral_dict,inplace=True)

# Add selectbox with a black patch behind the caption
st.sidebar.markdown("<div class='custom-selectbox'><p>Choose a Sector</p></div>", unsafe_allow_html=True)
industry = st.sidebar.selectbox('',all_data['Industry'].unique())


data = all_data[all_data['Industry']==industry]

st.sidebar.markdown("<div class='custom-selectbox'><p>Choose a Company</p></div>", unsafe_allow_html=True)
Company_Name = st.sidebar.selectbox('',data['Comp_Name'].unique())


Symbol = data[data['Comp_Name']==Company_Name]['Symbol'].values[0]
ticker_symbol = Symbol + '.NS'

################# Model for prediction of the sectoral indice close for next/current day ##########################

head_model = st.container(border=True)
head_model.markdown("#### <span style='color:#00838F'> {} Closing Forecast Model</span>".format(Company_Name),unsafe_allow_html=True)


#function to create appropriate partitions for the LSTM model
def partition_dataset(sequence_length, data, data2):
    x, y = [], []
    data_len = data.shape[0]
    index_Close = data2.columns.get_loc("Close")
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction
    
    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

#function to compute the technical indicators 
def calculate_technical_indicators(dat):
    period = 14
    smoothing_period = 3
    atr_period = 14

    # Stochastic Oscillator
    dat['Lowest Low'] = dat['Low'].rolling(window=period).min()
    dat['Highest High'] = dat['High'].rolling(window=period).max()
    dat['%K'] = ((dat['Close'] - dat['Lowest Low']) / (dat['Highest High'] - dat['Lowest Low'])) * 100
    dat['%D'] = dat['%K'].rolling(window=smoothing_period).mean()
    dat.drop(['Lowest Low', 'Highest High'], axis=1, inplace=True)

    # Average True Range (TR)
    dat['HL'] = dat['High'] - dat['Low']
    dat['HC'] = abs(dat['High'] - dat['Close'].shift(1))
    dat['LC'] = abs(dat['Low'] - dat['Close'].shift(1))
    dat['TR'] = dat[['HL', 'HC', 'LC']].max(axis=1)
    dat['ATR'] = dat['TR'].rolling(window=atr_period).mean()
    dat.drop(['HC', 'HL', 'LC', 'TR'], axis=1, inplace=True)

    # MACD And Signal Line
    dat['EMA_12'] = dat['Close'].ewm(span=12).mean()
    dat['EMA_26'] = dat['Close'].ewm(span=26).mean()
    dat['MACD'] = dat['EMA_12'] - dat['EMA_26']
    dat['Signal_Line'] = dat['MACD'].rolling(9).mean()
    dat.drop(['Open', 'High', 'Low', 'Adj Close'], axis=1, inplace=True)

    # RSI
    dat['Diff'] = dat['Close'].diff()
    dat['Gain'] = np.where(dat['Diff'] > 0, dat['Diff'], 0)
    dat['Loss'] = np.where(dat['Diff'] < 0, -dat['Diff'], 0)
    AL = dat['Loss'].rolling(14).mean()
    AG = dat['Gain'].rolling(14).mean()
    rs = AG / AL
    rsi = 100 - (100 / (1 + rs))
    dat['RSI'] = rsi
    dat.drop(['Diff', 'Gain', 'Loss'], axis=1, inplace=True)

    # Bollinger Bands
    dat['Upper_Band'] = dat['Close'].rolling(14).mean() + (2 * dat['Close'].rolling(14).std())
    dat['Lower_Band'] = dat['Close'].rolling(14).mean() - (2 * dat['Close'].rolling(14).std())

    return dat

#function to give a forecasted stock price
#the fucntion to combine the model and forecast the required output
@st.cache_resource
def stock_prediction(ticker):

    dat = yf.download(ticker)
    dat.drop('Volume',axis=1,inplace=True)
    dat.index = pd.to_datetime(dat.index)
    
    calculate_technical_indicators(dat)

    dat.dropna(inplace=True)
    
    df = dat.tail(504)
    
    # Indexing Batches
    train_df = df.sort_values(by=['Date']).copy()
    
    # List of considered Features
    FEATURES = df.columns.tolist()
    
    #head_model.text("Features Generated")

    # Create the dataset with features and filter the data to the list of FEATURES
    data = pd.DataFrame(train_df)
    data_filtered = data[FEATURES]

    # We add a prediction column and set dummy values to prepare the data for scaling
    data_filtered_ext = data_filtered.copy()
    data_filtered_ext['Prediction'] = data_filtered_ext['Close']

    
    # Get the number of rows in the data
    nrows = data_filtered.shape[0]

    # Convert the data to numpy values
    np_data_unscaled = np.array(data_filtered)
    np_data = np.reshape(np_data_unscaled, (nrows, -1))


    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = MinMaxScaler()
    np_data_scaled = scaler.fit_transform(np_data_unscaled)

    # Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = MinMaxScaler()
    df_Close = pd.DataFrame(data_filtered['Close'])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)
   
    # Set the sequence length - this is the timeframe used to make a single prediction
    sequence_length = 15

    # Prediction Index
    index_Close = data.columns.get_loc("Close")

    # Split the training data into train and train data sets
    # As a first step, we get the number of rows to train the model on 80% of the data 
    train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

    # Create the training and test data
    train_data = np_data_scaled[0:train_data_len, :]
    test_data = np_data_scaled[train_data_len - sequence_length:, :]
    
    # Generate training data and test data
    x_train, y_train = partition_dataset(sequence_length, train_data,data)
    x_test, y_test = partition_dataset(sequence_length, test_data,data)
    
    #head_model.text('Model is training')
    # Configure the neural network model
    model = Sequential()

    # Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
    n_neurons = x_train.shape[1] * x_train.shape[2]
   
    model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))) 
    model.add(LSTM(n_neurons, return_sequences=False))
    model.add(Dense(11))
    model.add(Dense(1))

    # Compile the model
    learning_rate = 0.001  # Example learning rate value
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # Training the model
    epochs = 100
    batch_size = 16
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs,
                        validation_data=(x_test, y_test)
                       )
  
    # Get the predicted values
    y_pred_scaled = model.predict(x_test)
    #head_model.text("Model is Generating the forecasted values")

    y_forcast_scaled = model.predict(x_test[-1:])

    # Unscale the predicted values
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_forcast = scaler_pred.inverse_transform(y_forcast_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))
    
    # Mean Absolute Percentage Error (MAPE)
    MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
    r2 = r2_score(y_pred,y_test_unscaled)
    return  y_forcast, MAPE,r2

# Create a thread to run the long-running function
#@st.cache_resource
#def run_long_running_function():
#    thread = threading.Thread(target=stock_prediction)
#    thread.start()

y_forcast,MAPE,r2 = stock_prediction(ticker_symbol)   


#function to plot the forecast output in a matplotlib chart 
@st.cache_resource
def model_predictor_graph(Ticker_Name,forecasted_value,mape):
    #importing required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    import matplotlib.dates as mdates
    import yfinance as yf
    
    df = yf.download(Ticker_Name)
    
    ##reading the data file to creat dates for forecast
    trading_days = pd.read_csv("Data/Trading_Days_2024.csv")
    current_day = df.tail(1).index[0]
    trading_days['Dates'] = pd.to_datetime(trading_days['Dates'])
    forecast_date = trading_days[trading_days['Dates']>current_day].iloc[0]['Dates']

    #woirking on first subplot
    plt.style.use('bmh')
    #setting the font size for the plot
    plt.rcParams['font.size'] = '16'
    #hfont = {'fontname':'Times New Roman'}
    plt.rcParams['font.style'] = 'italic'
    plt.rcParams['font.family'] = 'Times New Roman'
    #creating a subplot space and deciding the specs for the same
    fig,ax = plt.subplots(1,2,figsize=(20,7),gridspec_kw={'width_ratios':[1,1]})
    fig.tight_layout()
    #max of y (return values for the shaded box and line)
    ymax_ = df.tail(14)['Close'].max() #+ 0.05 * df.tail(14)['Close'].max()
    ymin_ = df.tail(14)['Close'].min() #+ 0.05 * df.tail(14)['Close'].min(
    #plotting the log normal returns of the index against the benchmark
    ax[0].plot(df.iloc[-252:,:]['Close'],label='Close')
    #ax[0].plot(df.set_index('POR').loc[:,['Benchmark_Return']],label='Benchmark_Return')
    ax[0].set_xlabel('Model Train Period',labelpad=10)
    ax[0].legend(framealpha=0)
    ax[0].set_title(f'Historical Prices',y=1.03)

    #creating border for the box
    ax[0].vlines(x=df.iloc[-14:,:]['Close'].index[0],ymin=ymin_,ymax=ymax_,ls='--',lw=3,color='black')
    ax[0].vlines(x=df.iloc[-252:,:]['Close'].index[-1],ymin=ymin_,ymax=ymax_,ls='--',lw=3,color='black')
    ax[0].hlines(y=ymax_,xmin=df.iloc[-14:,:]['Close'].index[0],xmax=df.iloc[-252:,:]['Close'].index[-1],ls='--',lw=3,color='black')
    ax[0].hlines(y=ymin_,xmin=df.iloc[-14:,:]['Close'].index[0],xmax=df.iloc[-252:,:]['Close'].index[-1],ls='--',lw=3,color='black')
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax[0].set_ylabel('Close')


    # Create blocked area in third axes
    xmax_ = df.iloc[-252:,:]['Close'].index[-1]
    xmin_ = df.iloc[-14:,:]['Close'].index[0]
    x_fill = [xmin_,xmin_,xmax_,xmax_]
    y_fill = [ymin_,ymax_,ymax_,ymin_]
    ax[0].fill(x_fill,y_fill,color='yellow',alpha=0.2,label='highlight')

    #creating the second zoomed in plot
    ax[1].set_facecolor('xkcd:eggshell')
    ax[1].plot(df.iloc[-3:,:]['Close'])
    ax[1].set_xlabel('Forcast',labelpad=10)
    ax[1].legend(framealpha=0)
    ax[1].set_title('Model Forecast',y=1.03)
    ax[1].set_xlim(df.iloc[-3:,:].index[0], df.tail(1).index + pd.Timedelta(days=5))
    # Set x-axis date format to 'yyyy-mm-dd'
    date_form = mdates.DateFormatter("%Y-%m-%d")
    ax[1].xaxis.set_major_formatter(date_form)
    #plotting a vertical line to show the summit day
    x_lim = ax[1].get_xlim()
    y_lim = ax[1].get_ylim()
    y_midpoint = np.mean(y_lim)
    upper_y = ymax_
    lower_y = ymin_

    forecasted_value = round(forecasted_value,2)
    upper_f = round(forecasted_value + 1*((mape/100) * forecasted_value),2)
    lower_f = round(forecasted_value - 1*((mape/100) * forecasted_value),2)

    #creating markers for forecasted values
    ax[1].scatter(x=df.tail(1).index,y=df.tail(1)['Close'] + (.015)*df.tail(1)['Close'] ,marker='^',s=140,color='black')
    ax[1].scatter(x=df.tail(1).index,y=df.tail(1)['Close'] - (.015)*df.tail(1)['Close'],marker='v',s=140,color='black')
    ax[1].scatter(x=forecast_date,y=forecasted_value,color='black',s=150,marker='d',label='Forecasted Value')
    ax[1].scatter(x=forecast_date,y=upper_f,color='green',s=150,marker='^',label='Forecasted Value Upper Bound')
    ax[1].scatter(x=forecast_date,y=lower_f,color='red',s=150,marker='v',label='Forecasted Value Lower Bound')
    ax[1].vlines(x=df.tail(1).index,ymin=df.tail(1)['Close'] - (.015)*df.tail(1)['Close'],ymax=df.tail(1)['Close'] + (.015)*df.tail(1)['Close'],ls='--',lw=3,color='black')
    ax[1].annotate(upper_f, (forecast_date,upper_f ), textcoords="offset points", xytext=(50,-2), ha='center')
    ax[1].annotate(lower_f, (forecast_date,lower_f ), textcoords="offset points", xytext=(50,-2), ha='center')
    ax[1].annotate(forecasted_value, (forecast_date,forecasted_value), textcoords="offset points", xytext=(50,-2), ha='center')
    ax[1].legend(loc='upper left')

    ax[1].fill([current_day,forecast_date,forecast_date], [df.tail(1)['Close'][0],forecasted_value,upper_f], color='green', alpha=0.3)  
    ax[1].fill([current_day,forecast_date,forecast_date], [df.tail(1)['Close'][0],forecasted_value,lower_f], color='red', alpha=0.3)  

    # Create bottom side of Connection patch for first axes
    con1 = ConnectionPatch(xyA=(max(ax[0].get_xlim())-14, ymin_), coordsA=ax[0].transData, 
                           xyB=(min(ax[1].get_xlim()),min(ax[1].get_ylim())), 
                           coordsB=ax[1].transData, color = 'black',arrowstyle='->')
    con1.set_linewidth(2)
    # Add bottom side to the figures
    fig.add_artist(con1)

    # Create upper side of Connection patch for first axes
    con2 = ConnectionPatch(xyA=(max(ax[0].get_xlim())-14, ymax_), coordsA=ax[0].transData, 
                           xyB=(min(ax[1].get_xlim()),max(ax[1].get_ylim())), 
                           coordsB=ax[1].transData, color = 'black',arrowstyle='->')
    con2.set_linewidth(2)
    # Add upper side to the figure
    fig.add_artist(con2)
    # Rotate the x-axis tick labels on subplot 0 for better readability
    plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45)
    plt.show() 
    return fig


col1_mod,col2_mod = st.columns([.8,.2])

mod_graph_space = col1_mod.container(border=True)
mod_graph_des_space = col2_mod.container(border=True,height=370)

mod_graph_space.pyplot(model_predictor_graph(ticker_symbol,y_forcast[0][0],MAPE))
mod_graph_des_space.markdown('<div style="text-align: justify; font-size: 12px">The model forecast is based onon technical indicators and features those that contrubute from a market sentiment perspective. The model forecasts the market close value for the current day while the market is open, and when it closed it forecasts the closing value of the next trading day. The model is trained on 504 past trading days with an R Squared value of {}. Our forecast tries to capture the market sentiments and the relevant price movements. While it is ok to take a reference from our model our suggestion is not to wholly rely on it.</div>'.format(round(r2,2)),unsafe_allow_html=True)

