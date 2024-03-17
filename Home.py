
#This is the home page
from errno import EDQUOT
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import os 
from PIL import Image
import base64
import streamlit as st

ico = Image.open("Graphics/Page Related Themes Etc/GT_Logo.png")

st.set_page_config(page_icon=ico,page_title='Homepage',layout='wide')

#creatin a function to convert an image into base64
@st.cache_data
def get_image_as_base64(file):
    with open(file,"rb") as f:
        data=f.read()
    return base64.b64encode(data).decode()

# Define CSS for image alignment

#Third element change is for making the arrow of the sidebar white
sidebar_style = """
<style>
    .st-emotion-cache-1rtdyuf {
        color: rgb(225, 225, 225);
        text-align: center;
    }
    
    .st-emotion-cache-1egp75f{
        color: rgb(225, 225, 225);
    }


    .st-emotion-cache-1pbsqtx{
    color:white;
    }


    .st-emotion-cache-j7qwjs.eczjsme7{
    color:white;
    }

    
</style>
"""
# Apply the CSS
st.markdown(sidebar_style, unsafe_allow_html=True)



#shuhsing the error prompt
st.set_option('deprecation.showPyplotGlobalUse', False)
#setting the page title and also the main icon


#getting the sidebar image
img = get_image_as_base64("Graphics/Page Related Themes Etc/Sidebar BG Edited.png")


# Set the favicon using base64 encoding
st.markdown(
    """
    <link rel="icon" href="data:image/png;base64,{},YOUR_BASE64_ENCODED_ICON">
    """.format(img),
    unsafe_allow_html=True
)

####################################### Managing the background attributes #######################################

# Define the CSS style with the GIF background
css = f'''<style> [data-testid="stAppViewContainer"] > .main {{
    opacity:1;
    background-image: url("https://media2.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy-downsized-large.gif");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    
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
    color: white;
    }}


    [data-testid="stVerticalBlockBorderWrapper"][class="st-emotion-cache-r421ms e1f1d6gn0"]{{
    background-color:rgba(0,0,0,.5);
    background-size:cover;
    }}


    </style>'''

# Use st.markdown() to inject the CSS for background and sidebar styling
st.write(css, unsafe_allow_html=True)


####################################### Creating the main section of the page #######################################


how = st.container(border=True)

# Define the HTML code for styling the title
title_html = """
    <style>
        /* CSS to change the color of the title */
        .title-text {
            color: white;text-align:center /* Change this to your desired color */
        }
    </style>
    <h3 class="title-text">PLUTUS: Profits Leveraged Using Tactical Understanding Strategies</h3>
"""

# Render the styled title using the HTML code
how.markdown(title_html, unsafe_allow_html=True)

#caption below the title
how.markdown(
    """
    <div style='text-align: center;color:white;font-size:18px;'><i>Welcome to Finance Frontier: Your Gateway to Financial Wisdom!</i></div>
    <hr style='border:2px solid #f0f0f0'>""",unsafe_allow_html=True)



# Working Steps section
how.markdown("""
    <div style='text-align: left;color:white;font-size:22px;'>How Does PLUTUS Work ?</div>""",unsafe_allow_html=True)
how.write("\n") 


# Define CSS for adjusting the size of list items
li_style = """
<style>
    .custom-li {
        font-size: 12px; /* Adjust the font size */
        /* Add any other CSS properties as needed */
    }
</style>
"""

# Apply the CSS
st.markdown(li_style, unsafe_allow_html=True)


working_steps = """
 📊 Data Collection:
   <ul><li style='font-size: 14px;'>Meticulously gathered historical stock prices, economic indicators, and sentiment data from reliable sources. We use the YahooFinance API for stock price data, Screener for fundamental data and Pulse by Zerodha for daily News.</li></ul>

 🛠️ Feature Engineering:
   <ul><li style='font-size: 14px;'>Our team identified key features, leveraging both technical indicators and market sentiment, to create powerful features that boost the acuracy and make the model more robust.</li></ul>

 🤖 Model Building:
   <ul><li style='font-size: 14px;'>Utilizing the power of Neural Networks and particularly the LSTM model, we built a robust predictive model that adapts to market dynamics and forecasts with much higher levels of accuarcy.</li></ul>

 📈 Evaluation:
   <ul><li style='font-size: 14px;'>Rigorous evaluation ensured our model's accuracy and performance, guaranteeing reliable predictions.</li></ul>
"""



col1, col2, col3, col4 = how.columns(4)

# Display content in columns
with col1:
    col1.markdown('''<div style='text-align: justify;color:white;font-size:16px;'>{}</div>'''.format(working_steps.split('\n\n')[0]),unsafe_allow_html=True)
with col2:
    col2.markdown('''<div style='text-align: justify;color:white;font-size:16px;'>{}</div>'''.format(working_steps.split('\n\n')[1]),unsafe_allow_html=True)
with col3:
    col3.markdown('''<div style='text-align: justify;color:white;font-size:16px;'>{}</div>'''.format(working_steps.split('\n\n')[2]),unsafe_allow_html=True)
with col4:
    col4.markdown('''<div style='text-align: justify;color:white;font-size:16px;'>{}</div>'''.format(working_steps.split('\n\n')[3]),unsafe_allow_html=True)
     
how.write('\n')     
#st.write('\n')
#st.write('\n')

# Relevance for Users section
how.markdown("""<div style='text-align: left;color:white;font-size:20px;'>What can Plutus do for you?</div>""",unsafe_allow_html=True)
how.write("\n")
how.markdown('''<div style='text-align: left;color:white;font-size:14px;'>💪 Empower Your Decisions: Make informed investment decisions based on precise stock price predictions.</div>''',unsafe_allow_html=True)
how.markdown('''<div style='text-align: left;color:white;font-size:14px;'>🛡️ Risk Management: Understand market sentiment for effective risk mitigation.</div>''',unsafe_allow_html=True)
how.markdown('''<div style='text-align: left;color:white;font-size:14px;'>🌐 Holistic Insights: Gain a comprehensive understanding of factors influencing stock prices.</div>''',unsafe_allow_html=True)
how.write("\n")
how.write("\n")
#st.markdown(relevance_section)



team_space = st.container(border=True)
# Add some colors and icons
team_space.markdown(
    """
    <h3 style='color: #FFFFFF; text-align: center;'><i>Meet The Team Behind PLUTUS</i></h3>
    <hr style='border:2px solid #FFFFFF'>
    """,
    unsafe_allow_html=True,
)


####################################### Creating the profiles section #######################################

# Open the image file
image_edwin = Image.open("Graphics/Profile Pictures/Edwin_Joice.png")
image_shambhavi = Image.open("Graphics/Profile Pictures/Shambhavi_Mishra.png")
image_akansha = Image.open("Graphics/Profile Pictures/Akansha_Choudhary.png")
image_jubin = Image.open("Graphics/Profile Pictures/Jubin_Joseph.png")

#Defining column structure for the contributors
col1,col2,col3,col4 = team_space.columns(4)
edwin_space = col1.container(border=True)
shabhavi_space = col2.container(border=True)
akansha_space = col3.container(border=True)
jubin_space = col4.container(border=True)

#inserting the images first
edwin_space.image(image_edwin, use_column_width=True)
shabhavi_space.image(image_shambhavi, use_column_width=True)
akansha_space.image(image_akansha, use_column_width=True)
jubin_space.image(image_jubin, use_column_width=True)

#Names for the profiles
edwin_space.markdown('''<div style="text-align: center;font-size: 16px;color:white">{}</div>'''.format('Edwin Joice'),unsafe_allow_html=True)
shabhavi_space.markdown('''<div style="text-align: center;font-size: 16px;color:white">{}</div>'''.format('Shambhavi Mishra'),unsafe_allow_html=True)
akansha_space.markdown('''<div style="text-align: center;font-size: 16px;color:white">{}</div>'''.format('Akansha Choudhary'),unsafe_allow_html=True)
jubin_space.markdown('''<div style="text-align: center;font-size: 16px;color:white">{}</div>'''.format('Jubin Joseph'),unsafe_allow_html=True)

#Designations for the profiles
edwin_space.markdown('''<div style="text-align: center;font-size: 14px;color:white">{}</div>'''.format('NLP Expert'),unsafe_allow_html=True)
shabhavi_space.markdown('''<div style="text-align: center;font-size: 14px;color:white">{}</div>'''.format('Subject Matter Expert'),unsafe_allow_html=True)
akansha_space.markdown('''<div style="text-align: center;font-size: 14px;color:white">{}</div>'''.format('Subject Matter Expert'),unsafe_allow_html=True)
jubin_space.markdown('''<div style="text-align: center;font-size: 14px;color:white">{}</div>'''.format('ML Engineer'),unsafe_allow_html=True)

#Mail Ids
edwin_space.markdown('''<div style="text-align: center;font-size: 14px;color:white">{}</div>'''.format('edwin.joice@in.gt.com'),unsafe_allow_html=True)
shabhavi_space.markdown('''<div style="text-align: center;font-size: 14px;color:white">{}</div>'''.format('shambhavi.mishra@in.gt.com'),unsafe_allow_html=True)
akansha_space.markdown('''<div style="text-align: center;font-size: 14px;color:white">{}</div>'''.format('akansha.choudhary@in.gt.com'),unsafe_allow_html=True)
jubin_space.markdown('''<div style="text-align: center;font-size: 14px;color:white">{}</div>'''.format('jubin.joseph@in.gt.com'),unsafe_allow_html=True)

#adding an extra line for space
edwin_space.write('\n')
shabhavi_space.write('\n')
akansha_space.write('\n')
jubin_space.write('\n')


# Define CSS for image alignment
image_style = """
<style>
    .image-container-right {
        display: flex;
        justify-content: flex-end; /* Align images to the right */
        align-items: center; /* Align images vertically at the center */
    }
    
    .image-container-left {
        display: flex;
        justify-content: flex-start; /* Align images to the left */
        align-items: center; /* Align images vertically at the center */
    }

    .image-container-center {
        display: flex;
        justify-content: center; /* Align images to the left */
        align-items: center; /* Align images vertically at the center */
    }
</style>
"""

# Apply the CSS
st.markdown(image_style, unsafe_allow_html=True)

#encoding the logos
link_test = get_image_as_base64("Graphics/Logos/L_Logo.png")
git_test = get_image_as_base64("Graphics/Logos/G_Logo.png")

#Creating Two Columns for the Linked and GitHub Logos and also hyperlinking them (edwin)
ed1,ed2 = edwin_space.columns(2)
with ed1:
    st.markdown("<div class='image-container-right'><a href={} target='_blank'><img src='data:image/png;base64,{}'width='36'></div>".format('https://www.linkedin.com/in/edwinjoice/',link_test), unsafe_allow_html=True)

with ed2:
    st.markdown("<div class='image-container-left'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='38'></div>".format('https://github.com/Edwin-Joice',git_test), unsafe_allow_html=True)
edwin_space.write('\n')

#Creating Two Columns for the Linked and Git Hub Logos and also hyperlinking them (shambhavi)
sh1,sh2 = shabhavi_space.columns(2)
with sh1:
    st.markdown("<div class='image-container-right'><a href={} target='_blank'><img src='data:image/png;base64,{}'width='36'></div>".format('https://www.linkedin.com/in/shambhavi-mishra-bb248617a/',link_test), unsafe_allow_html=True)

with sh2:
    st.markdown("<div class='image-container-left'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='38'></div>".format('https://github.com/mshambhavi2000',git_test), unsafe_allow_html=True)
shabhavi_space.write('\n')


#Creating Two Columns for the Linked and Git Hub Logos and also hyperlinking them (akansha)
ak1,ak2 = akansha_space.columns(2)
with ak1:
    st.markdown("<div class='image-container-right'><a href={} target='_blank'><img src='data:image/png;base64,{}'width='36'></div>".format('https://www.linkedin.com/in/akansha-choudhary-300a091a6/',link_test), unsafe_allow_html=True)

with ak2:
    st.markdown("<div class='image-container-left'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='38'></div>".format('https://github.com/aka-hash',git_test), unsafe_allow_html=True)
akansha_space.write('\n')

#Creating Two Columns for the Linked and Git Hub Logos and also hyperlinking them (jubin)
jj1,jj2 = jubin_space.columns(2)
with jj1:
    st.markdown("<div class='image-container-right'><a href={} target='_blank'><img src='data:image/png;base64,{}'width='36'></div>".format('https://www.linkedin.com/in/jubin-joseph10/',link_test), unsafe_allow_html=True)

with jj2:
    st.markdown("<div class='image-container-left'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='38'></div>".format('https://github.com/JubinJoseph-10',git_test), unsafe_allow_html=True)
jubin_space.write('\n')



####################################### Acknowledgement Section for all the required companies #######################################
credits_space = st.container(border=True)

credits_space.markdown("""
    <div style='text-align: center;color:white;font-size:24px;'>Acknowledgements</div>""",unsafe_allow_html=True)

credits_space.write('\n')
credits_space.write('\n')

col_1,col_2,col_3,col_4 = credits_space.columns(4)

#encoding the logos for display
screener_logo = get_image_as_base64("Graphics/Logos/Screener_Logo.png")
pulse_logo = get_image_as_base64("Graphics/Logos/Pulse Logo.png")
yfin_logo = get_image_as_base64("Graphics/Logos/Yahoo!_Finance_logo_2021.png")
nse_logo = get_image_as_base64("Graphics/Logos/NSE_Logo.png")
ibef_logo = get_image_as_base64("Graphics/Logos/IBEF_Logo1.png")
inv_logo = get_image_as_base64("Graphics/Logos/Inv.com_logo1.png")
MW4M = get_image_as_base64("Graphics/Logos/MW4m_logo1.png")
Mon_Con = get_image_as_base64("Graphics/Logos/Mon_Con_logo1.png")


#writting acks for screener
with col_1:
    st.markdown("<div class='image-container-center'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='200'></div>".format('https://www.screener.in/',screener_logo), unsafe_allow_html=True)
    st.write('\n')
    st.markdown('''<div style="text-align: justify;font-size: 14px;color:white">{}</div>'''.format('All the data for our findamental analysis has been taken from Screener. We would like to acknowledge and express our gratitude for providing valuable data that contributes to the accuracy and robustness of our project\'s insights.'),unsafe_allow_html=True)
#writting acks for pulse
with col_2:
    st.markdown("<div class='image-container-center'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='200'></div>".format('https://pulse.zerodha.com/',pulse_logo), unsafe_allow_html=True)
    st.write('\n')
    st.markdown('''<div style="text-align: justify;font-size: 14px;color:white">{}</div>'''.format('All the news articles for the sentiment analysis and the latest articles on a sector or company have been scrapped from Pulse by Zerodha. We are grateful for their application that allows us to access the latest newws from the market.'),unsafe_allow_html=True)    
#writting acks for yahoofinance
with col_3:
    st.markdown("<div class='image-container-center'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='125'></div>".format('https://finance.yahoo.com/',yfin_logo), unsafe_allow_html=True)
    st.write('\n')
    st.markdown('''<div style="text-align: justify;font-size: 14px;color:white">{}</div>'''.format('The widely used Yahoo Finance API allowed us to access a wide range of stock price data that we have leveraged further for our model. Yahoo Finance API allows us to fetch and work with the data really efficiently'),unsafe_allow_html=True)
#writting acks for nseindia
with col_4:
    st.markdown("<div class='image-container-center'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='200'></div>".format('https://www.nseindia.com/',nse_logo), unsafe_allow_html=True)
    st.write('\n')
    st.markdown('''<div style="text-align: justify;font-size: 14px;color:white">{}</div>'''.format('We not only took inspiration from the idea of Nifty Sectoral indices but also used the list of constituents which NSE updated monthly.'),unsafe_allow_html=True)
#IBEF acki
with col_1:
    st.write('\n')
    st.write('\n')
    st.markdown("<div class='image-container-center'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='80'></div>".format('https://www.ibef.org/',ibef_logo), unsafe_allow_html=True)
    st.write('\n')
    st.markdown('''<div style="text-align: justify;font-size: 14px;color:white">{}</div>'''.format('We took inspiration for the inroduction of the sectoral indices from the articles in IBEF (India Brand Equity Foundation).'),unsafe_allow_html=True)
    st.write('\n')
    st.write('\n')
#investing.com acki
with col_2:
    st.write('\n')
    st.write('\n')
    st.markdown("<div class='image-container-center'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='200'></div>".format('https://www.investing.com/',inv_logo), unsafe_allow_html=True)
    st.write('\n')
    st.markdown('''<div style="text-align: justify;font-size: 14px;color:white">{}</div>'''.format('We took data for some of the sectoral indices from Investing.com.'),unsafe_allow_html=True)    
    st.write('\n')
    st.write('\n')
#MoneyWorks4me acki
with col_3:
    st.write('\n')
    st.write('\n')
    st.markdown("<div class='image-container-center'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='170'></div>".format('https://www.moneyworks4me.com/',MW4M), unsafe_allow_html=True)
    st.write('\n')
    st.markdown('''<div style="text-align: justify;font-size: 14px;color:white">{}</div>'''.format('We took inspiration from a few of the elements of the Web Application of MoneyWorks4Me.'),unsafe_allow_html=True)    
    st.write('\n')
    st.write('\n')
#money control acki
with col_4:
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.markdown("<div class='image-container-center'><a href={} target='_blank'><img src='data:image/png;base64,{}' width='170'></div>".format('https://www.moneycontrol.com/',Mon_Con), unsafe_allow_html=True)
    st.write('\n')
    st.markdown('''<div style="text-align: justify;font-size: 14px;color:white">{}</div>'''.format('We took inspiration from some elements of the Web Application of MoneyControl.'),unsafe_allow_html=True)    
    st.write('\n')
    st.write('\n')
