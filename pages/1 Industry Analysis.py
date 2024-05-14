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
import base64



ico = Image.open("Graphics/Page Related Themes Etc/GT_Logo.png")
st.set_page_config(page_icon=ico,page_title='Sectoral Analyis',layout='wide')


# Define CSS for image alignment
sidebar_style = """
<style>
    .st-emotion-cache-1rtdyuf {
        color: rgb(225, 225, 225);
        text-align: center;
    }
    
    .st-emotion-cache-1egp75f{
        color: rgb(225, 225, 225);
    }

        .element-container st-emotion-cache-lark4g e1f1d6gn4{
        align-items: center;
    }

    .st-emotion-cache-79elbk.eczjsme10 span {
        color: white;
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
    background-color:rgba(0,0,0,.50);
    background-size:cover;
    }}


    [data-testid="stSidebarNavCollapseIcon"] {{
    display: none !important;
    }}    


    </style>'''



# Use st.markdown() to inject the CSS for background and sidebar styling
st.write(css, unsafe_allow_html=True)


all_data = pd.read_csv("Data/All_Data_Industries.csv")

sectoral_dict = {'niftyautolist' : 'Nifty AUTO' , 'niftybanklist' : 'Nifty BANK', 'niftyconsumerdurableslist':'Nifty CONSUMER DURABLES',
       'niftyfinancelist':'Nifty FINANCIAL SERVICES', 'niftyfmcglist' :'Nifty FMCG', 'niftyhealthcarelist':'Nifty HEALTHCARE',
       'niftyitlist':'Nifty IT', 'niftymedialist': 'Nifty MEDIA', 'niftymetallist': 'Nifty METAL',
       'niftyoilgaslist':'Nifty OIL & GAS', 'niftypharmalist':'Nifty PHARMA', 'niftyrealtylist':'Nifty REALTY'}

all_data.replace(sectoral_dict,inplace=True)
all_data.set_index('Date',inplace=True)

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
st.sidebar.markdown("<div class='custom-selectbox'><p>Choose a Sectoral Indice</p></div>", unsafe_allow_html=True)

# Display the selectbox
Sectoral_Indice = st.sidebar.selectbox('', all_data['Industry'].unique())



yfin_ticker_sectoral = {'Nifty AUTO' : '^CNXAUTO' , 'Nifty BANK':'^NSEBANK', 'Nifty CONSUMER DURABLES':'^CNXCONSUM',
       'Nifty FINANCIAL SERVICES':'NIFTY_FIN_SERVICE.NS', 'Nifty FMCG':'^CNXFMCG', 'Nifty HEALTHCARE':'NIFTY_HEALTHCARE.NS',
       'Nifty IT':'^CNXIT', 'Nifty MEDIA':'^CNXMEDIA','Nifty METAL':'^CNXMETAL',
       'Nifty OIL & GAS':'NIFTY_OIL_AND_GAS.NS', 'Nifty PHARMA':'^CNXPHARMA', 'Nifty REALTY':'^CNXREALTY'}


#Displaying the title of the page
st.write("# <span style='color:#6700A5'>{}</span>".format(Sectoral_Indice),unsafe_allow_html=True)
st.divider()


############################ LINE CHART WITH THE SECTORAL INDICE AND THE DESCRIPTION ####################################
#displaying the price trend for the sectoral indice
#first section with the line plot and the introduction
#Fetching the logos for images display
indice_image_names = [f for f in os.listdir("Graphics/Indice Images") if f.endswith(('.png', '.jpg', '.jpeg'))]
# Create a list to store the images


image_list = []


# Load each image and append it to the list
for name in indice_image_names:
    image_path = os.path.join("Graphics/Indice Images", name)
    img = Image.open(image_path)
    image_list.append(img)

try:
    position = indice_image_names.index(Sectoral_Indice + ".png")
except ValueError:
    try:
        position = indice_image_names.index(Sectoral_Indice + ".jpg")
    except ValueError:
        # Handle the case when the ".jpg" file is not found either
        print(f"Error: {Sectoral_Indice}.jpg not found.")


col1,col2 ,col3 = st.columns(spec=[.24,0.595,0.15],)

with col3:
    box_3= col3.container(height=419)
    
    option = box_3.radio(label="",options=['All Time','10 Yrs','5 Yrs','3 Yrs','1 Yr','6 Months','1 Month'])


with col2:
    box_2 = col2.container(border=True)
    #box_rad = col2.container(border =True)
    #option = box_rad.radio(label="",options=['All Time','10 Yrs','5 Yrs','3 Yrs','1 Yr','6 Months','1 Month'],horizontal=True)
    selected_options = {'All Time':None,'10 Yrs':'10y','5 Yrs':'5y',
                        '3 Yrs':'3y','1 Yr':'1y','6 Months':'6mo','1 Month':'1mo'}
    data = yf.download(yfin_ticker_sectoral[Sectoral_Indice],period = selected_options[option])['Close']
    fig = px.line(data,title=f'{Sectoral_Indice} Price Trend',y='Close',labels={"value": f"{Sectoral_Indice} Close", "index": "Date"},line_shape="linear",height = 380)
    box_2.plotly_chart(fig,use_container_width=True)

    
#industry descriptions
industry_descriptions = {'Nifty AUTO': 'The Nifty Auto sector encompasses companies manufacturing and distributing automobiles and related products. Examples include passenger and commercial vehicles, two-wheelers, and auto components. The Sector historically mirrors the country\'s economic health, driven by a strong two-wheeler segment due to a growing middle-class and youth population. Future growth is expected through trends like vehicle electrification.' , 
 'Nifty BANK': 'The Nifty Bank sector is a dynamic segment essential to the financial ecosystem, facilitating transactions, savings, and investments. Operating as financial institutions, banks play a crucial role in everyday economic activities, offering services such as savings accounts, loans, and investments. With a focus on customer transactions, risk management, and financial intermediation, banks cater to diverse needs. Reforms such as digital payments, neo-banking, and the rise of Indian non-banking financial companies (NBFCs) have significantly boosted financial inclusion and spurred the credit cycle.', 
 'Nifty CONSUMER DURABLES': 'The Nifty Consumer Durables sector comprises a range of everyday products characterized by durability and consumer demand. These goods, including appliances, electronics, and household items, are known for their long life and essential role in daily living. Companies in the Nifty Consumer Durables sector prioritize product durability, brand development, and effective distribution networks. Growing disposable income and technological innovation in India are fueling heightened demand for consumer durable goods, intensifying competition among brands.', 
 'Nifty FINANCIAL SERVICES': 'The Financial Services sector constitutes a broad category encompassing a range of financial products and solutions essential for individuals and businesses. These services, including banking, insurance, and investment products, are fundamental to economic transactions, risk management, and wealth creation. In recent years, India has witnessed substantial growth in financial services, with exports in the banking and financial sectors seeing an uptrend. Projections indicate a robust trajectory, with the Indian financial services sector expected to further contribute to the country\'s GDP and play a crucial role in economic development.', 
 'Nifty FMCG': 'Fast-moving consumer Goods (FMCG) are everyday products known for their rapid turnover, low cost, and high consumer demand. Examples include food, personal care items, and cleaning products. FMCG companies focus on high sales volumes, brand building, and efficient distribution networks. With a global presence and resilience in economic downturns, the FMCG sector plays a vital role in meeting essential needs and driving economic activity.', 
 'Nifty HEALTHCARE': 'The Nifty Healthcare sector spans a diverse range of essential products and services vital to public well-being. It encompasses various components like hospitals, medical devices, clinical trials, outsourcing, telemedicine, medical tourism, health insurance, and medical equipment. This sector is characterized by its crucial role in delivering medical solutions, enhancing overall health, and responding to the evolving needs of the population.', 
 'Nifty IT': 'The Nifty IT sector is a vital component within the financial landscape, playing a key role in technology-driven services, solutions, and innovations. It encompasses companies involved in information technology, software development, and related services. Nifty IT companies contribute to the digital transformation of businesses, offering solutions in areas such as software development, cybersecurity, and cloud computing. The Nifty IT sector is crucial for enhancing organizational efficiency and competitiveness. Its global presence and adaptability to changing technological landscapes position it as a key growth catalyst for India\'s economy.',
 'Nifty MEDIA': 'The Nifty Media sector stands as a pivotal force in the economic landscape, representing a sunrise segment that is making significant strides. This sector is driven by factors such as widespread access to fast and affordable internet, rising incomes, and increased consumer spending on media and entertainment. Nifty Media is characterized by its unique attributes compared to other markets, with a notable emphasis on high volumes and a rising Average Revenue Per User (ARPU). The Nifty Media sector is poised for substantial growth, fueled by increasing consumer demand and a positive trajectory in advertising revenue.', 
 'Nifty METAL': 'The Nifty Metal sector is known for its rapid turnover, low cost, and high consumer demand. Nifty Metal, comprising various sub-sectors, is characterized by its focus on high production volumes, brand positioning, and efficient supply chain networks. With a wide range of products, including base metals, steel, and alloys, the Nifty Metal sector caters to diverse industrial demands. The expansion in the industry is propelled by the abundant domestic supply of raw materials and affordable labor. The industry\'s emphasis on production efficiency and strategic branding enhances its market position and global competitiveness.', 
 'Nifty OIL & GAS': 'Nifty Oil and Gas, consisting of various sub-sectors, emphasizes high production volumes, brand development, and efficient supply chain networks. With a diverse range of products, including petroleum, natural gas, and refined products, the Nifty Oil and Gas sector caters to diverse industrial and domestic demands. The industry\'s focus on operational efficiency and strategic branding enhances its market position and global competitiveness. Given the strong link between India\'s economic growth and energy demand, the sector is poised for increased investment opportunities.',
 'Nifty PHARMA': 'The Nifty Pharma sector is globally acknowledged as a significant contributor to the pharmaceutical landscape. India is globally recognized as a major provider of affordable generic drugs and vaccines. The pharmaceutical industry in India has experienced significant growth, ranking third in pharmaceutical production by volume. Nifty Pharma is recognized for its robust performance and growth, driven by factors such as research and development, drug manufacturing, and healthcare solutions. Its efficiency in delivering essential pharmaceutical products plays a vital role in sustaining public health and driving economic activity.', 
 'Nifty REALTY': 'The Nifty Realty sector is a globally recognized segment comprising housing, retail, hospitality, and commercial sub-sectors. The sector\'s growth is intricately linked with the corporate environment\'s expansion and the demand for office space, urban, and semi-urban accommodations. It holds a prominent position, ranking among the key contributors to the economy. Its growth is among the 14 major sectors in the economy, the construction industry holds the third position, considering its direct, indirect, and induced effects.'}

with col1:
    
    box_1 = col1.container(border=True,height=419)
    box_1.image(image_list[position])
    box_1.markdown('<div style="text-align: justify; font-size: 13px">{}</div>'.format(industry_descriptions[Sectoral_Indice]), unsafe_allow_html=True)



############################ TABLE WITH OVERALL INVESTMENT SCORES ####################################
###section for displaying the peer to peer analysis tables
#Financial Year Creator Function
def Financial_Year_Creator(df):
    df['FY'] = df.index.astype(str).tolist()
    df['Financial_Year'] = df['FY'].apply(lambda x : (int(x[:4])+1) if x[5:7] =='12' else (int(x[:4])))
    df.drop(columns='FY',inplace=True)


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
word_list = word_search[Sectoral_Indice].tolist() 



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
gauge_space.markdown('###### {} Sentiment Gauge'.format(Sectoral_Indice))
gauge_space.plotly_chart(fig,use_container_width=True) 
find_more = col4.container(border=True)
find_more.markdown("<div style='display: flex; justify-content: center; align-items: center; text-align: center; font-size: 18px;'><a href={} target='_blank'>For the Latest Scoop, Click Here!</div>".format('https://pulse.zerodha.com/'), unsafe_allow_html=True)
find_more.write('\n')


################# Overall Industry Constituents Ratings ##########################
#RATIOS FOR OVR RATING
list_rat = {'ROCE':'ROCE','ROE_5yr_cagr':'ROE_5yr_cagr',
 'Debt_to_Equity':'Debt_to_Equity','Interest_Cov_Rat':'Interest_Cov_Rat',
 'FixAsset_Turn_Rat':'FixAsset_Turn_Rat','Inv_Turn_Rat':'Inv_Turn_Rat',
 'P/E':'P/E','EV/EBITDA':'EV/EBITDA','EV/Sales':'EV/Sales','CFO/Sales':'CFO/Sales',
'CFO/Total Debt':'CFO/Total Debt'}

#class for company rates
class company_rater_industry_analysis:
    
    #attributes required in the whole function
    def __init__(self,FY,Industry,List_of_ratios):
        self.FY = FY
        self.Industry = Industry 
        self.List_of_ratios = List_of_ratios
    
    #the ratings shall be for the latest year and based on a particular indutry
    def sectoral_components_extract(FY,Industry,data):
        Financial_Year_Creator(data)
        Latest_Slice = data[data['Financial_Year']==2023]
        Industry_Slice = Latest_Slice[Latest_Slice['Industry']==Industry]
        Industry_Slice.replace({np.inf:5,-np.inf:5},inplace=True)
        return Industry_Slice
    
    ## Functions for giving scores 
    #function to return value of under valued/overvalued shares
    def margin_caller_val_de(median,x,margin,upper_margin):
        if (median-margin)<=x<=(median+margin):
            return 3
        elif x<=median-upper_margin:
            return 5
        elif x>=median+upper_margin:
            return 1
        elif (median-upper_margin)<=x<=(median-margin):
            return 4
        elif (median+margin)<=x<=(median+upper_margin):
            return 2
        else:
            return np.nan
    
    
    ###Function to create a new column for valuation
    def valuation_computer_val_de(ratio,df):
        mid = df[ratio].median()
        mar = (df[ratio].max()-df[ratio].min())*(.10)
        upper_mar = (df[ratio].max()-df[ratio].min())*(.20)
        new_col_name = f'{ratio} Based Valuation'
        df[new_col_name] = df[ratio].apply(lambda x:company_rater_industry_analysis.margin_caller_val_de(mid,x,mar,upper_mar))

    #function to return value of under valued/overvalued shares
    def margin_caller_oth(median,x,margin,upper_margin):
        if (median-margin)<=x<=(median+margin):
            return 3
        elif x>=median+upper_margin:
            return 5
        elif x<=median+upper_margin:
            return 1
        elif (median-upper_margin)<=x<=(median-margin):
            return 2
        elif (median+margin)<=x<=(median+upper_margin):
            return 4
        else:
            return np.nan

    ###Function to create a new column for valuation
    def profitability_scorer(ratio,df):
        mid = df[ratio].median()
        mar = (df[ratio].max()-df[ratio].min())*(.10)
        upper_mar = (df[ratio].max()-df[ratio].min())*(.20)
        new_col_name = f'{ratio} Based Score'
        df[new_col_name] = df[ratio].apply(lambda x:company_rater_industry_analysis.margin_caller_oth(mid,x,mar,upper_mar))
    
    ###Function to create a new column for valuation
    def scorer(Industry_Slice,List_of_ratios):
        for rat in List_of_ratios:
            if rat in ['Debt_to_Equity','EV/EBITDA','EV/Sales','P/E']:
                company_rater_industry_analysis.valuation_computer_val_de(List_of_ratios[rat],Industry_Slice)
            else:
                company_rater_industry_analysis.profitability_scorer(List_of_ratios[rat],Industry_Slice)
                
    def valuation_displayer(data):
        data['Valuation'] = round((data['EV/Sales Based Valuation'] + data['EV/EBITDA Based Valuation'] + data['P/E Based Valuation'])/3)
        data['Valuation'].replace({1:'Overvalued',2:'Somewhat Overvalued',3:'Fairly Valued',4:'Somewhat Undervalued',5:'Undervalued'},inplace=True)
    
    def weigh_score_assigner(data,weight_dict):
        data['Investment_Score'] = data['ROCE Based Score'].fillna(1).astype('float64') * weight_dict['ROCE Based Score'] + data['ROE_5yr_cagr Based Score'].fillna(1).astype('float64') * weight_dict['ROE_5yr_cagr Based Score'] + data['Debt_to_Equity Based Valuation'].fillna(1).astype('float64') * weight_dict['Debt_to_Equity Based Valuation'] + data['Interest_Cov_Rat Based Score'].fillna(1).astype('float64') * weight_dict['Interest_Cov_Rat Based Score'] + data['FixAsset_Turn_Rat Based Score'].fillna(1).astype('float64') * weight_dict['FixAsset_Turn_Rat Based Score'] + data['Inv_Turn_Rat Based Score'].fillna(1).astype('float64') * weight_dict['Inv_Turn_Rat Based Score'] + data['P/E Based Valuation'].fillna(1).astype('float64') * weight_dict['P/E Based Valuation'] + data['EV/EBITDA Based Valuation'].fillna(1).astype('float64') * weight_dict['EV/EBITDA Based Valuation'] + data['EV/Sales Based Valuation'].fillna(1).astype('float64') * weight_dict['EV/Sales Based Valuation'] + data['CFO/Sales Based Score'].fillna(1).astype('float64') * weight_dict['CFO/Sales Based Score'] + data['CFO/Total Debt Based Score'].fillna(1).astype('float64') * weight_dict['CFO/Total Debt Based Score']
        data['Investment_Score'] = round(data['Investment_Score'])

#for weight dict
weights = pd.read_csv("Data/Weights for Overall Score.csv")
weights.set_index('Index_Name',inplace=True)
weights = weights.astype('float64')
weights_dict = {}
for x in weights.columns:
    weights_dict[x] = weights[x].loc[Sectoral_Indice]


#Relevant Slice
req = company_rater_industry_analysis.sectoral_components_extract(2023,Sectoral_Indice,all_data)

#Scoring the Ratios
company_rater_industry_analysis.scorer(req,list_rat)

#Table Columns for Valuation Metrics
company_rater_industry_analysis.valuation_displayer(req)

#changing the dtype of the ratios
req[['Operating Margin','ROCE','ROE_5yr_cagr','Sales_5yr_cagr','Profit_5yr_cagr','Debt_to_Equity']] = req[['Operating Margin','ROCE','ROE_5yr_cagr','Sales_5yr_cagr','Profit_5yr_cagr','Debt_to_Equity']].astype('float64')

#computing the overall investment score based on the weight assiigned
company_rater_industry_analysis.weigh_score_assigner(req,weights_dict)

#putting everything in a single frame which we will display
display_frame=pd.DataFrame()
display_frame.index = req['Comp_Name']
req.set_index('Comp_Name',inplace=True)
display_frame['Investment_Score'] = req['Investment_Score'].replace({1:'⭐',2:'⭐⭐',3:'⭐⭐⭐',4:'⭐⭐⭐⭐',5:'⭐⭐⭐⭐⭐'})
display_frame[['Operating Margin (%)','ROCE (%)','ROE_5yr_cagr (%)','Sales_5yr_cagr (%)','Profit_5yr_cagr (%)','Debt_to_Equity (%)']] = round(req[['Operating Margin','ROCE','ROE_5yr_cagr','Sales_5yr_cagr','Profit_5yr_cagr','Debt_to_Equity']]*100,2)
display_frame['P/E'] = round(req['P/E'].astype('float64'),2)
display_frame['Valuation'] = req['Valuation']


#Company Investment Ratings Space
col1,col2 = st.columns([.25,.75])

ratings_space_1 = col1.container(border=True)
ratings_space_1.write("<span style='color:#00838F'>Ratings Description Here:</span>",unsafe_allow_html=True)
ratings_space_1.markdown('<div style="text-align: justify; font-size: 12px">This table presents investment ratings derived from the analysis of financial ratios for various companies in their respective industries. The ratings are determined using the latest financial year\'s data, with each financial ratio assigned a weight according to the industry in which the company operates. These weighted ratios provide insights into the financial health and performance of the companies, aiding in investment decision-making. <br><br> Note of Caution: It\'s important to recognize that the ratings presented here are a result of our analysis and may reflect our opinions and perspectives. Your own analysis and opinions may differ based on different interpretations of the data and varying investment strategies.</div>',unsafe_allow_html=True)
ratings_space_2 = col2.container(border=True,height=477)
ratings_space_2.dataframe(display_frame.fillna('N.A'),use_container_width=True,height=455)


############################ FETCHING DATA FOR RELATIVE GROWTH AND RISK RETURN PLOT ####################################

#retreiving the tickers and company name 
rename_ticker = {}
for x in req['Symbol']+'.NS':
    rename_ticker[x] = req[req['Symbol']==x[:-3]].index[0]

#Definiion for bringing data
req_data = yf.download(list(rename_ticker.keys()),period='5y')
req_close = req_data['Close'].dropna()



############################ RISK RETURN ANALYSIS ####################################

##function to compute annual risk and return
@st.cache_resource
def risk_return_computer(sector_close,index_name):
    daily_ret_risk = sector_close.pct_change().dropna()
    annual_risk_return = daily_ret_risk.agg(['mean','std']).T
    annual_risk_return['Annual Return (%)'] = (annual_risk_return['mean'] * (252) * 100).round(2)
    annual_risk_return['Annual Risk (%)'] = (annual_risk_return['std'] * np.sqrt(252) * 100).round(2)
    annual_risk_return.drop(columns=['mean','std'],inplace=True)
    annual_risk_return.index.name = index_name
    annual_risk_return.rename(index=rename_ticker,inplace=True)
    return annual_risk_return

##function to compute annual risk and return
fmcg_risk_ret = risk_return_computer(req_close,Sectoral_Indice)


def risk_return_plotter(risk_return_df,industry_name):
    fig = px.scatter(data_frame=risk_return_df,x='Annual Risk (%)',y='Annual Return (%)',color=risk_return_df.index,title=f'Risk Return Analysis of Constituents of {Sectoral_Indice}')

    vertical_line_value = risk_return_df['Annual Risk (%)'].median()
    horizontal_line_value = risk_return_df['Annual Return (%)'].median()
    
    #adding the industry mean line for risk and return
    fig.add_shape(
        go.layout.Shape(type='line',
            x0=vertical_line_value,
            x1=vertical_line_value,
            y0=risk_return_df['Annual Return (%)'].min(),
            y1=risk_return_df['Annual Return (%)'].max(),
            line=dict(color='red', width=2,dash='dot')))

    fig.add_shape(go.layout.Shape(type='line',
            x0=risk_return_df['Annual Risk (%)'].min(),
            x1=risk_return_df['Annual Risk (%)'].max(),
            y0=horizontal_line_value,
            y1=horizontal_line_value,
            line=dict(color='blue', width=2,dash='dot')))


    # Adding annotations to explain industry mean line
    fig.add_annotation(go.layout.Annotation(text='Industry Median Annual Risk',
            x=vertical_line_value,
            y=risk_return_df['Annual Return (%)'].max(),
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40))

    fig.add_annotation(go.layout.Annotation(text='Industry Median Annual Return',
            x=risk_return_df['Annual Risk (%)'].max(),
            y=horizontal_line_value,
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=+40))

    return fig

#creating the instance plot
fmcg_risk_ret.index.name = Sectoral_Indice
fmcg = risk_return_plotter(fmcg_risk_ret,'Nifty_FMCG')

#plotting the risk return scatter plot
col2,col1 = st.columns(spec=[.75,.25])
col1_rr = col1.container(border=True,height=489)
col2_rr = col2.container(border=True)

col2_rr.plotly_chart(fmcg,use_container_width=True)
col1_rr.write("<span style='color:#00838F'>Risk Return Analysis</span>",unsafe_allow_html=True)
col1_rr.markdown('<div style="text-align: justify; font-size: 12px">The Risk vs Return scatter helps segment the companies into Low/ High Risk and Return. This interactive plot can assess all companies in the particular sector based on their risk-return profile. With an increasing risk, the potential return you might get from a company might also increase. The return component represents the annual return of that stock. This has been calculated by multiplying average daily returns with the number of trading days, i.e., 252 in a year. The risk component is the standard deviation which is the volatility or variability of returns. This has been calculated by multiplying the standard deviation of daily returns by the square root of the number of trading days.</div>',unsafe_allow_html=True)


############################ RELATIVE GROWTH ####################################
col_1,col_2 = st.columns([.25,.75])

#space for graph and text
cons_rel_1 = col_1.container(border=True,height=489)
cons_rel_2 = col_2.container(border=True) 

cons_rel_1.write("<span style='color:#00838F'>Relative Growth Analysis</span>",unsafe_allow_html=True)
cons_rel_1.markdown('<div style="text-align: justify; font-size: 12px">This graph can help in identifying the overall trend of the sector whether it is increasing, decreasing, or fluctuating. The relative growth of closing prices has been normalized by dividing each closing price with its initial closing price. This has been done to represent change relative to the initial closing price. This is why the initial value presented is 100. Underperforming or outperforming stocks can also be analyzed in terms of growth. If any stock is showing unusual spikes or dips in relative growth, it might require further investigation. Changes in relative growth might also coincide with certain economic events, company announcements during that period, and so on.</div>',unsafe_allow_html=True)

@st.cache_resource
def cl_pr_ret(req_close):
    closing_price_df = req_close.div(req_close.iloc[0]).mul(100)
    closing_price_df.rename(columns=rename_ticker,inplace=True)
    closing_price_df.columns.name = Sectoral_Indice
    return closing_price_df
closing_price_df =  cl_pr_ret(req_close)

fig = px.line(closing_price_df, title="RELATIVE GROWTH", labels={"value": "Growth Multiple", "index": "Date"},line_shape="linear", render_mode="svg")
cons_rel_2.plotly_chart(fig,use_container_width=True)

################# Relative Growth With Nifty 50 ##########################
col_1,col_2 = st.columns([.80,.20])

growth_with_benchmark_space = col_1.container(border=True)
analysis_description_space = col_2.container(border=True,height=489)


comp_indices_data = pd.DataFrame()
Tickers = [yfin_ticker_sectoral[Sectoral_Indice],'^NSEI']
for x in Tickers:
    comp_indices_data[x] = yf.download(x,period = '5y')['Adj Close']
    norm_indices_df = comp_indices_data.div(comp_indices_data.iloc[0]).mul(100)
norm_indices_df.rename(columns={'^NSEI':'Nifty 50',yfin_ticker_sectoral[Sectoral_Indice]:Sectoral_Indice},inplace=True)    
norm_indices_df.columns.name = f"Relative Performance of {Sectoral_Indice}"
fig = px.line(norm_indices_df, title=f"Nifty50 -{Sectoral_Indice}", labels={"value": "Close"},line_shape="linear", render_mode="svg")

growth_with_benchmark_space.plotly_chart(fig,use_container_width=True)
analysis_description_space.write("<span style='color:#00838F'>Relative Growth Analysis between {} & Nifty 50</span>".format(Sectoral_Indice),unsafe_allow_html=True)
analysis_description_space.markdown('<div style="text-align: justify; font-size: 12px">The following graph provides a comparison of the performance between Nifty 50 and the particular industry. This can be an insight into the correlation, growth, return, and market trends of both indices over the past 5 years. If for example, the index trend is above Nifty 50 (which is a snapshot of the overall market), it might indicate outperformance, superior sectoral strength, and investor confidence in the specific industry.</div>',unsafe_allow_html=True)

################# Model for prediction of the sectoral indice close for next/current day ##########################

head_model = st.container(border=True)
head_model.markdown("#### <span style='color:#00838F'> {} Closing Forecast Model</span>".format(Sectoral_Indice),unsafe_allow_html=True)


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


#the fucntion to combine the model and forecast the required output
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

y_forcast,MAPE,r2 = stock_prediction(yfin_ticker_sectoral[Sectoral_Indice])   

#function to plot the forecast output in a matplotlib chart 
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
    upper_f = round(forecasted_value + 2*((mape/100) * forecasted_value),2)
    lower_f = round(forecasted_value - 2*((mape/100) * forecasted_value),2)

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

mod_graph_space.pyplot(model_predictor_graph(yfin_ticker_sectoral[Sectoral_Indice],y_forcast[0][0],MAPE))
mod_graph_des_space.markdown('<div style="text-align: justify; font-size: 12px">The model forecast is based onon technical indicators and features those that contrubute from a market sentiment perspective. The model forecasts the market close value for the current day while the market is open, and when it closed it forecasts the closing value of the next trading day. The model is trained on 504 past trading days with an R Squared value of {}. Our forecast tries to capture the market sentiments and the relevant price movements. While it is ok to take a reference from our model our suggestion is not to wholly rely on it.</div>'.format(round(r2,2)),unsafe_allow_html=True)
mod_graph_des_space.write('\n')


