#creating a page for any stock price prediction

#importing all the requires libraries
from httpx import head
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
import base64

ico = Image.open("Graphics/Page Related Themes Etc/GT_Logo.png")
st.set_page_config(page_icon=ico,page_title='Company Analyis',layout='wide')

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

#########################Function to fetch the financial year as a column#########################
def Financial_Year_Creator(df):
    df['FY'] = df.index.astype(str).tolist()
    df['Financial_Year'] = df['FY'].apply(lambda x : (int(x[:4])+1) if x[5:7] =='12' else (int(x[:4])))
    df.drop(columns='FY',inplace=True)

Financial_Year_Creator(all_data)

##Few Changes
all_data = all_data.round(4)
all_data["ROE"] = (all_data["ROE"] * 100).round(2)
all_data["ROCE"] = (all_data["ROCE"] * 100).round(2)

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

# Add selectbox with a black patch behind the caption
st.sidebar.markdown("<div class='custom-selectbox'><p>Choose a Sector</p></div>", unsafe_allow_html=True)
industry = st.sidebar.selectbox('',all_data['Industry'].unique())


data = all_data[all_data['Industry']==industry]

st.sidebar.markdown("<div class='custom-selectbox'><p>Choose a Company</p></div>", unsafe_allow_html=True)
company = st.sidebar.selectbox('',data['Comp_Name'].unique())

title_html = """
    <h2 class="title-text">{}</h2>
""".format(company)

st.markdown(title_html,unsafe_allow_html=True)
st.write(all_data[all_data['Comp_Name']==company]['Comp_Description'][0])
st.divider()
st.write('\n')
################# Latest News and Sentiment Gauge ##########################

# Global variable to hold the DataFrame
all_news_df = pd.DataFrame()
#scrapper function to get the news data quickly
def News_Scrapper():
    #importing modeuls and getting the link
    global all_news_df
    import pandas as pd
    from bs4 import BeautifulSoup
    import requests
    import datetime
    url = 'https://pulse.zerodha.com/'
    r = requests.get(url)
    soup = BeautifulSoup(r.content) 
    #CREATING A VRAIBLES FOR THE ARTCILE SPACE
    article_space = soup.find('ul',{'id':'news'})
    all_article = article_space.find_all('li',{'class':'box item'})
    #creating a comprehensise data_frame with the articles
    data = pd.DataFrame()
    title,links,bodies,sources_,report_time,report_day = [],[],[],[],[],[]
    for art in all_article:
        titi = art.find('h2',{'class':'title'}) 
        tit = art.find('h2',{'class':'title'}).text
        link = titi.find('a')['href'] 
        bod = art.find('div',{'class':'desc'}).text
        source = art.find('span',{'class':'feed'}).text
        time_day = art.find('span',{'class':'date'})['title'].split(',')
        title.append(tit)
        links.append(link)
        bodies.append(bod)
        sources_.append(source)
        report_time.append(time_day[0])
        report_day.append(time_day[1])
    data['Article_Title'] = title
    data['Article_Link'] = links
    data['Article_Bodies'] = bodies
    data['Article_Source'] = sources_
    data['Article_Time'] = report_time
    data['Article_Date'] = report_day
    data['TimeStamp'] = pd.to_datetime(data['Article_Time'] + data['Article_Date'])
    data['Seconds_Ago'] = (datetime.datetime.now() - data['TimeStamp']).apply(lambda x:x.seconds)
    data['Minutes_Ago'] = round(data['Seconds_Ago'] / 60)
    data['Hours_Ago'] = round(data['Seconds_Ago'] / 3600,2)
    data['Days_Ago'] = (datetime.datetime.now() - data['TimeStamp']).apply(lambda x:x.days)
    
    #fcuntion to compute when the news was posted
    def time_ago_computer(df_with_cols):
        ago=[]
        for i in range(len(df_with_cols)):
            if data.iloc[:,-3:].values[:,-1][i] >= 1:
                if data.iloc[:,-3:].values[:,-2][i] >= 1:
                    time_ago = str(data.iloc[:,-3:].values[:,-1][i]).split('.')[0] + ' Day ' + str(data.iloc[:,-3:].values[:,-2][i].split('.')[0]).split('.')[0] + ' Hours ' + str(round(float('.'+str(data.iloc[:,-3:].values[:,-2][i]).split('.')[1])*60))  + ' Minutes'
                    ago.append(time_ago)
                else:
                    time_ago = str(data.iloc[:,-3:].values[:,-1][i]).split('.')[0] + ' Day ' + str(data.iloc[:,-3:].values[:,-1][i]).split('.')[0] + ' Minutes'
                    ago.append(time_ago)
            else:
                if data.iloc[:,-3:].values[:,-2][i] >= 1:
                    time_ago = str(data.iloc[:,-3:].values[:,-2][i]).split('.')[0] + ' Hours ' +  str(round(float('.'+str(data.iloc[:,-3:].values[:,-2][i]).split('.')[1])*60))  + ' Minutes'
                    ago.append(time_ago)
                else:
                    time_ago =  str(data.iloc[:,-3:].values[:,-3][i]).split('.')[0] + ' Minutes'
                    ago.append(time_ago)
        return ago

    #computing the time posted ago
    data['Time_Ago'] = time_ago_computer(data) 
    
    all_news_df = data

News_Scrapper()


# all_news+df is the global variables holding all the new articles

#Extracting the derived keywords 
word_search = pd.read_csv("Data/Words search.csv")
word_list = word_search[industry].tolist() 



#function to fetch data based on relevance
#news_file.fillna('_',inplace=True)
def filter_func(data, word_list):
    temp = data.copy()
    temp['Article_Bodies'] = temp['Article_Bodies'].apply( lambda x:x.lower())
    temp['Article_Title']  = temp['Article_Title'].apply( lambda x:x.lower())
    filtered_data = pd.DataFrame()# Initialize an empty DataFrame with the same columns as 'data'
    for i in range(data.shape[0]):  # Loop through each row in 'data'
        sentence = ' '.join(temp.iloc[i][['Article_Title', 'Article_Bodies']].values)# Concatenate the article title and bodies into a single string
        words_in_sentence = sentence.split()  # Split the sentence into words
        for word_to_search in word_list:  # Loop through each word in 'word_list'
            if word_to_search in words_in_sentence:  # Check if the word is in the list of words in the sentence
                filtered_data = pd.concat([filtered_data,pd.DataFrame(data.iloc[i]).T],axis=0) # Append the matching row to 'filtered_data'
                break  # Break out of the inner loop if a match is found to avoid appending duplicates
    return filtered_data


#structuring tje section for news and sentiments analysis
col1,col2,col3,col4  = st.columns(spec=[.24,0.24,0.24,.28],)

df_news = filter_func(all_news_df,word_list)
req_frame = df_news.iloc[:3,:]

def displayer(req_frame):
    for col,(x,row) in zip([col1,col2,col3],req_frame.iterrows()):
        section = col.container(border=True,height=376)
        link = row['Article_Link']
        title = row['Article_Title']
        body = row['Article_Bodies']
        source = row['Article_Source']
        time_ago = row['Time_Ago']
        section.markdown("###### *[{}]({})*".format(title,link),unsafe_allow_html=True)
        section.markdown("<span style='font-size: 12px; line-height:0.8;'>{}</span>".format(body), unsafe_allow_html=True)
        section.markdown('''<div style="text-align: left;font-size: 10px">{}</div>'''.format(source) + '''<div style="text-align: left;font-size: 12px">{}</div>'''.format(time_ago),unsafe_allow_html=True)

# Function to display articles
displayer(req_frame)


nltk.download("stopwords")


port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub("[^a-zA-z]",' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words("english")]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content


def stem_and_concatenate(dataframe, title_column, body_column):
    stemmed_data = []  # Initialize an empty list to store stemmed data
    stemmer = PorterStemmer()
    
    for i in range(dataframe.shape[0]):  # Loop through each row in the dataframe
        title = dataframe.iloc[i][title_column]  # Get the title from the specified column
        body = dataframe.iloc[i][body_column]  # Get the body from the specified column
        sentence = ' '.join([title, body])  # Concatenate the title and body into a single string
        stem = ' '.join([stemmer.stem(word) for word in sentence.split()])  # Apply stemming to the concatenated string
        stemmed_data.append(stem)  # Append the stemmed string to the list
    
    dataframe["stemmed_data"] = stemmed_data  # Create a new column in the dataframe with the stemmed data

stem_and_concatenate(df_news,'Article_Title','Article_Bodies')

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()


def vader_sentiment_scores(data):
    # Calculate sentiment scores for each stemmed sentence
    data['sentiment_scores'] = data['stemmed_data'].apply(lambda x: sid.polarity_scores(x))

    # Extract and store positive, negative, and neutral scores in separate columns
    data['V_positive'] = data['sentiment_scores'].apply(lambda x: x['pos'])*10
    data['V_neutral'] = data['sentiment_scores'].apply(lambda x: x['neu'])*10
    data['V_negative'] = data['sentiment_scores'].apply(lambda x: x['neg'])*10


    # Map the compound score to a scale of -10 to 10
    data['v_compound_sentiment'] = data['sentiment_scores'].apply(lambda x: x['compound'])*10
    return data

vader_sentiment_scores(df_news)



plot_bgcolor = "#FFFFFF"
quadrant_colors = [plot_bgcolor, "#2bad4e", "#85e043", "#eff229", "#f2a529", "#f25829"]
quadrant_text = ["","<b>Positive</b>", "<b></b>", "<b>Neutral</b>", "<b></b>", "<b>Negative</b>"]
n_quadrants = len(quadrant_colors) - 1

current_value = int(df_news['v_compound_sentiment'].mean())
min_value = -10
max_value = 10
hand_length = np.sqrt(2) / 4
hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

fig = go.Figure(
    data=[
        go.Pie(
            values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
            rotation=90,
            hole=0.5,
            marker_colors=quadrant_colors,
            text=quadrant_text,
            textinfo="text",
            hoverinfo="skip",
        ),
    ],
    layout=go.Layout(
        showlegend=False,
        margin=dict(b=0,t=5,l=5,r=5),
        width=225,
        height=225,
        paper_bgcolor=plot_bgcolor,
        annotations=[
            go.layout.Annotation(
                text=f"<b>Sentiment Score (-10 to 10):</b><br>{current_value} units",
                x=0.5, xanchor="center", xref="paper",
                y=0.25, yanchor="bottom", yref="paper",
                showarrow=False,
            ),
            go.layout.Annotation(
                text="<b>The sentiment scores are</b><b>derived from the <br> latest news </b><b>and analyzed using the</b><b><br>VADER sentiment analysis model.</b>",
                x=0.5, y=0.05, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10),
            )
            
        ],
            
        shapes=[
            go.layout.Shape(
                type="circle",
                x0=0.48, x1=0.52,
                y0=0.48, y1=0.52,
                fillcolor="#333",
                line_color="#333",
            ),
            go.layout.Shape(
                type="line",
                x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                line=dict(color="#333", width=4)
            )
        ]
    )
)
fig.update_traces(textfont_size=9)
gauge_space = col4.container(border=True,height=305)
gauge_space.markdown('###### {} Sentiment Gauge'.format(industry))
gauge_space.plotly_chart(fig,use_container_width=True) 
find_more = col4.container(border=True)
find_more.markdown("<div style='display: flex; justify-content: center; align-items: center; text-align: center; font-size: 18px;'><a href={} target='_blank'>For the Latest Scoop, Click Here!</div>".format('https://pulse.zerodha.com/'), unsafe_allow_html=True)
find_more.write('\n')


##########################Creating Metrics for the top of the page##########################

st.divider()

#latest fy
fy = data[data['Comp_Name']==company]['Financial_Year'].tail(1)[0]

# Define the HTML code for styling the title
metrics_title_html = f"""
    <style>
        /* CSS to change the color of the title */
        .title-text {{
            color: black;text-align:left /* Change this to your desired color */
        }}
    </style>
    <h4 class="title-text">Key Metrics of {company} for the Financial Year {fy}</h4>
"""

# Render the styled title using the HTML code
st.markdown(metrics_title_html, unsafe_allow_html=True)


head_metrics = st.columns(5)

#1. Net Profit
Net_Profit = data[data['Comp_Name']==company]['Net profit'].tail(1).to_list()[0]
Net_Profit_Growth = (((data[data['Comp_Name']==company]['Net profit'].tail(2)[1]/data[data['Comp_Name']==company]['Net profit'].tail(2)[0]) - 1)*100).round(2)

#2. EPS
EPS = int(data[data['Comp_Name']==company]['EPS'].tail(1).to_list()[0])
EPS_Growth = (((data[data['Comp_Name']==company]['EPS'].tail(2)[1]/data[data['Comp_Name']==company]['EPS'].tail(2)[0]) - 1)*100).round(2)

#3. ROE
ROE = (data[data['Comp_Name']==company]['ROE'].tail(1)[0]).round(2) 
ROE_Growth = (((data[data['Comp_Name']==company]['ROE'].tail(2)[1]/data[data['Comp_Name']==company]['ROE'].tail(2)[0]) - 1)*100).round(2)

#4. P/E
P_E = data[data['Comp_Name']==company]['P/E'].tail(1)[0].round(2)   
#P_E_Growth = (((data[data['Comp_Name']==company]['P/E'].tail(2)[1]/data[data['Comp_Name']==company]['P/E'].tail(2)[0]) - 1)*100).round(2)

#4. P/E
Market_Cap = data[data['Comp_Name']==company]['Market Cap'].tail(1)[0].round(2)   
#Market_Cap_Growth = (((data[data['Comp_Name']==company]['Market Cap'].tail(2)[1]/data[data['Comp_Name']==company]['Market Cap'].tail(2)[0]) - 1)*100).round(2)


#creating the lists with metrics and their values
met_title = ['Net Profit (Rs Crores)','Earning Per Share','ROE (%)','P/E','Market Cap (Rs Crores)']
head_met = [Net_Profit,EPS,ROE,P_E,Market_Cap]
head_met_trend = [Net_Profit_Growth,EPS_Growth,ROE_Growth,0,0]

########################## Displaying metrics at the top of the Page##########################
for col,met in zip(head_metrics,range(5)):
    tile = col.container(border=True)
    tile.metric(label=met_title[met],value=head_met[met],delta=head_met_trend[met])




################################# COMPARISON CHARTS ######################################
plot_area = st.container(border=True)    
c1,c2= plot_area.columns((4,7))

c1.write("<span style='color:#00838F'>Comparative analysis between Industry Constituents</span>",unsafe_allow_html=True)
c1.markdown('<div style="text-align: justify; font-size: 14px">Comparing two companies can often be tricky,one way to look at it is comparing peers or companies that are part of the same industry. The comparitive company analysis allows you to see not only how the company you have invested in has performed over the year given a metric of your choice, but also allows you to compare such growth and change with respect to is peers and competitors. The choice of metric also plays a huge role in the investment process. Since its not only a reflection of your investment perspective but also a barometer for your expectations as an investor.</div>',unsafe_allow_html=True)
c1.write('\n')
c1.write('\n')

c1_con = c1.container(border=True)

with c1_con:
    selected_ratio = st.selectbox("Select a Metric:", ['Net profit','Sales',
                                                    'EBITDA','ROE','ROCE','EPS','COGS'])
    options = data['Comp_Name'].unique()
    selected_options = st.selectbox(f"Select a Company to compare with {company}", options)
    plot_company = selected_options
    plot_metrics = selected_ratio
    default_columns=  ["Comp_Name", "Financial_Year"]
    all_columns = default_columns +[plot_metrics]
    plot_data = data[all_columns]
    filtered_plot_data = plot_data[plot_data["Comp_Name"].isin([plot_company,company])]

    
sns.set_style("whitegrid")  # sns.set_style("ticks")

# Create the Plotly bar plot
fig = px.bar(
    filtered_plot_data[filtered_plot_data['Financial_Year']>=2016],
    x='Financial_Year',
    y=plot_metrics,
    color='Comp_Name',
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    labels={'Comp_Name': 'Company', 'Financial_Year': 'Year', plot_metrics: 'Metric'},
)

# Customize the layout
fig.update_layout(
    xaxis_title='Year',
    yaxis_title=plot_metrics,
    legend=dict(title='Company', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

chart_space = c2.container(border=True)
# Display the plot in Streamlit
chart_space.plotly_chart(fig,use_container_width=True)


##########################Displaying growth metrics at the Bottom of the Page##########################
growth_metrics_space = st.container(border=True)

growth_metrics_space.write("<span style='color:#00838F;font-size:18px'>Diving Deeper Into The Growth Metrics</span>",unsafe_allow_html=True)

Profit,Sales = growth_metrics_space.columns(2)
ROE,Stock_Price = growth_metrics_space.columns(2)

##########################Displaying growth metrics at the Bottom of the Page##########################
Symbol = data[data['Comp_Name']==company]['Symbol'].values[0]
#1. Growth Metrics (Sales)
Sales_5yr_CAGR = (data[data['Comp_Name']==company]['Sales_5yr_cagr'].tail(1)[0] * 100).round(2)
Sales_3yr_CAGR = (data[data['Comp_Name']==company]['Sales_3yr_cagr'].tail(1)[0] * 100).round(2)
Sales_Year_Growth = (data[data['Comp_Name']==company]['Yearly_Sales_Growth'].tail(1)[0] * 100).round(2)
Sales_Frame = pd.DataFrame()
Sales_Frame.index = ['Sales 5yr','Sales 3yr','Yearly Sales Growth']
Sales_Frame.index.name = Symbol + '_Sales'
Sales_Frame['CAGR (%)'] = [Sales_5yr_CAGR,Sales_3yr_CAGR,Sales_Year_Growth]

#2. Growth Metrics (Net profit)
Net_Profit_5yr_CAGR = (data[data['Comp_Name']==company]['Profit_5yr_cagr'].tail(1)[0] * 100).round(2)
Net_Profit_3yr_CAGR = (data[data['Comp_Name']==company]['Profit_3yr_cagr'].tail(1)[0] * 100).round(2)
Net_Profit_Year_Growth = (data[data['Comp_Name']==company]['Yearly_Profit_Growth'].tail(1)[0] * 100).round(2)
Profit_Frame = pd.DataFrame()
Profit_Frame.index = ['Profit CAGR 5yr','Profit CAGR 3yr','Yearly Profit Growth']
Profit_Frame.index.name = Symbol + '_Profit'
Profit_Frame['CAGR (%)'] = [Net_Profit_5yr_CAGR,Net_Profit_3yr_CAGR,Net_Profit_Year_Growth]


#3. Growth Metrics (ROE)
ROE_5yr_CAGR = (data[data['Comp_Name']==company]['ROE_5yr_cagr'].tail(1)[0] * 100).round(2)
ROE_3yr_CAGR = (data[data['Comp_Name']==company]['ROE_3yr_cagr'].tail(1)[0] * 100).round(2)
Yearly_ROE_Growth = (data[data['Comp_Name']==company]['Yearly_ROE_Growth'].tail(1)[0] * 100).round(2)
ROE_Frame = pd.DataFrame()
ROE_Frame.index = ['ROE 5yr','ROE 3yr','ROE Yearly Growth']
ROE_Frame.index.name = Symbol + '_ROE'
ROE_Frame['CAGR (%)'] = [ROE_5yr_CAGR,ROE_3yr_CAGR,Yearly_ROE_Growth]


#4. Growth Metrics (Stock Price)
tick =  Symbol +'.NS'
stock_data_5 = yf.download(tick,period='5y')
stock_data_3 = yf.download(tick,period='3y')
stock_data_1 = yf.download(tick,period='1y')
Stock_Price_CAGR_5yr = ((((stock_data_5.tail(1)['Close'][0]/stock_data_5.head(1)['Close'][0])**(1/5))-1)* 100).round(2)
Stock_Price_CAGR_3yr = ((((stock_data_3.tail(1)['Close'][0]/stock_data_3.head(1)['Close'][0])**(1/3))-1) * 100).round(2)
Stock_Price_Yearly_Growth = (((stock_data_1.tail(1)['Close'][0]/stock_data_1.head(1)['Close'][0]) - 1) * 100).round(2)
Stock_Price_Frame = pd.DataFrame()
Stock_Price_Frame.index = ['Stock Price 5yr','Stock Price 3yr','Stock Price Yearly Growth']
Stock_Price_Frame.index.name = Symbol + '_Stock_Price'
Stock_Price_Frame['CAGR (%)'] = [Stock_Price_CAGR_5yr,Stock_Price_CAGR_3yr,Stock_Price_Yearly_Growth]


#Creating a list of dataframes
Labels = ['Profit CAGR','Sales CAGR','ROE CAGR','Stock Price CAGR']
CAGR_frames = [Profit_Frame,Sales_Frame,ROE_Frame,Stock_Price_Frame] 


Tile_Profit = Profit.container(border=True)
Tile_Sales = Sales.container(border=True)
Tile_ROE= ROE.container(border=True,height=287)
Tile_Stock_Price = Stock_Price.container(border=True)

col_Prof_Des,col_Prof = Tile_Profit.columns(2)
col_Sales_Des,col_Sales = Tile_Sales.columns(2)
col_ROE,col_ROE_Des = Tile_ROE.columns(2)
col_Stock_Price,col_Stock_Price_Des = Tile_Stock_Price.columns([.55,.45])


col_Prof.dataframe(CAGR_frames[0])
col_Prof_Des.write("<span style='color:#00838F'>Profit based CAGR</span>",unsafe_allow_html=True)
col_Prof_Des.markdown('<div style="text-align: justify; font-size: 12px">Analyzing the Compound Annual Growth Rate (CAGR) values over the past 5 years and 3 years provides valuable insight into the financial performance of the company. A consistent and positive CAGR in profit/net profit signifies robust financial health and operational efficiency. This metric reflects the company\'s ability to generate sustainable earnings over time, which is crucial for long-term investors seeking stable returns on their investment.</div>',unsafe_allow_html=True)
col_Prof_Des.write('\n')


col_Sales.dataframe(CAGR_frames[1])
col_Sales_Des.write("<span style='color:#00838F'>Sales based CAGR</span>",unsafe_allow_html=True)
col_Sales_Des.markdown('<div style="text-align: justify; font-size: 12px">Examining the CAGR values for sales over the past 5 years and 3 years offers key insights into the company\'s revenue growth trajectory. A steadily increasing CAGR in sales indicates the company\'s ability to expand its market presence, capture market share, and drive top-line growth. For investors, a healthy CAGR in sales is indicative of the company\'s ability to capitalize on market opportunities and deliver value to shareholders.</div>',unsafe_allow_html=True)
col_Sales_Des.write('\n')


col_ROE.dataframe(CAGR_frames[2])
col_ROE_Des.write("<span style='color:#00838F'>ROE based CAGR</span>",unsafe_allow_html=True)
col_ROE_Des.markdown('<div style="text-align: justify; font-size: 12px">Understanding the CAGR values for Return on Equity (ROE) over the past 5 years and 3 years provides valuable insights into the company\'s profitability and efficiency in utilizing shareholder equity. A consistently high and growing ROE CAGR reflects the company\'s ability to generate strong returns on shareholder investments. Investors often view a positive trend in ROE CAGR as a sign of effective management and sustainable growth potential.</div>',unsafe_allow_html=True)
col_ROE_Des.write('\n')


col_Stock_Price.dataframe(CAGR_frames[3])
col_Stock_Price_Des.write("<span style='color:#00838F'>Stock Price based CAGR</span>",unsafe_allow_html=True)
col_Stock_Price_Des.markdown('<div style="text-align: justify; font-size: 12px">Evaluating the CAGR values for the company\'s stock price over the past 5 years and 3 years is essential for investors seeking capital appreciation. A positive and steady CAGR in stock price signifies investor confidence, market performance, and potential for capital gains. By analyzing historical trends in stock price CAGR, investors can make informed decisions regarding the company\'s growth prospects and investment opportunities.</div>',unsafe_allow_html=True)
col_Stock_Price_Des.write('\n')


################# Model for prediction of the sectoral indice close for next/current day ##########################

head_model = st.container(border=True)
head_model.markdown("#### <span style='color:#00838F'> {} Closing Forecast Model</span>".format(company),unsafe_allow_html=True)


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

y_forcast,MAPE,r2 = stock_prediction(tick)   


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


col1_mod,col2_mod = head_model.columns([.77,.23])

mod_graph_space = col1_mod.container(border=True)
mod_graph_des_space = col2_mod.container(border=True)

mod_graph_space.pyplot(model_predictor_graph(tick,y_forcast[0][0],MAPE))
mod_graph_des_space.markdown('<div style="text-align: justify; font-size: 12px">The model forecast is based onon technical indicators and features those that contrubute from a market sentiment perspective. The model forecasts the market close value for the current day while the market is open, and when it closed it forecasts the closing value of the next trading day. The model is trained on 504 past trading days with an R Squared value of {}. Our forecast tries to capture the market sentiments and the relevant price movements. While it is ok to take a reference from our model our suggestion is not to wholly rely on it.</div>'.format(round(r2,2)),unsafe_allow_html=True)
mod_graph_des_space.write('\n')

