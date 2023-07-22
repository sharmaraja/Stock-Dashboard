
# Imports

import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews

import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

# Dashboarding
# Taking Details of Stock
st.title('Stock Dashboard')
ticker=st.sidebar.text_input('Ticker')
start_date=st.sidebar.date_input('Start Date')
end_date=st.sidebar.date_input('End Date')

# Downloading and ploting the stock
data=yf.download(ticker,start=start_date,end=end_date)
fig=px.line(data,x=data.index, y=data['Adj Close'], title=ticker,labels='Daily Chart')
st.plotly_chart(fig)


# Making tabs for further details
pricing_data, fundamental_data, news = st.tabs(['Pricing Data','Fundamental Data', 'News'])

# Pricing Details
with pricing_data:
    st.header('Price Movements')
    df_mod=data.copy()
    df_mod['% Change'] = data['Adj Close']/data['Adj Close'].shift(1)-1
    df_mod.dropna(inplace=True)
    st.write(df_mod)
    anual_return=df_mod['% Change'].mean()*252*100
    std_dev=np.std(df_mod['% Change']*np.sqrt(252))
    st.write('Anual return is ', round(anual_return,2),'%')
    st.write('Standard Deviation is ', round(std_dev*100,2),'%')
    st.write('Risk Adj return is ',round(anual_return/(std_dev*100),2))

# Fundamental Details of Stock
with fundamental_data:
    key = 'EXC5E416K37YAJDK'
    fd = FundamentalData(key,output_format = 'pandas')
                         
    st.subheader("Balance Sheet")
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list (balance_sheet.T.iloc[0])
    st.write(bs)
                         
    st.subheader(' Income Statement')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)

    st.subheader("Cash Flow Statement")
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)

    
# Top 10 news
with news:
    st.header(f'News of {ticker}')
    sn = StockNews(ticker , save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
#         st.write(df_news['published'[i]])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')