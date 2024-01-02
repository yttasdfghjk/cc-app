import streamlit as st
from streamlit_autorefresh import st_autorefresh

import cryptocompare
import pandas as pd
import numpy as np
from datetime import datetime

import plotly.graph_objects as go

from prophet import Prophet
from prophet.plot import plot_plotly

count = st_autorefresh(interval=3600000, key="refreshcounter")

tickers = ["BTC", "ETH", "MATIC", "ADA", "DOT", "SOL", "LINK", "CAKE",
         "VET", "ICP", "AVAX", "FTM", "FET", "RNDR", "SEI", "SUI"]

currency = 'USDT'
pricesDfListDaily =[]
pricesDfListHourly =[]
limit_value = 2000
exchange_name = 'Binance'


def create_candlestickchart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['datetimes'], 
        open=df['open'], 
        high=df['high'], 
        low=df['low'], 
        close=df['close']) )
    
    return fig

#def getPriceListData(timeframe, tickers, pricesDfList):
def getPriceListData(timeframe, tickers):

    pricesDfList =[]
    #pricesDfListHourly =[]
    limit_value = 2000
    exchange_name = 'Binance'
    
    year = int(pd.Timestamp.today().strftime('%Y'))
    month = int(pd.Timestamp.today().strftime('%m'))
    day = int(pd.Timestamp.today().strftime('%d'))
    hour = int(pd.Timestamp.today().strftime('%H'))
    toDate = datetime(year, month, day, 0, 0)
    toHour = datetime(year, month, day, hour, 0)

    for t in tickers:
    # Define the ticker symbol and other details
        ticker_symbol = t

        # For daily data
        if timeframe=="daily":
            df=cryptocompare.get_historical_price_day(ticker_symbol, currency, limit=limit_value, exchange=exchange_name, toTs=toDate)
        elif timeframe=="hourly":
            df=cryptocompare.get_historical_price_hour(ticker_symbol, currency, limit=limit_value, exchange=exchange_name, toTs=toHour)

        df = pd.DataFrame(df)
        df.set_index("time", inplace=True)
        df.index = pd.to_datetime(df.index, unit='s')
        df['datetimes'] = df.index
        if timeframe=="daily":
            df['datetimes'] = df['datetimes'].dt.strftime('%Y-%m-%d')
        elif timeframe=="hourly":
            df['datetimes'] = df['datetimes'].dt.strftime('%Y-%m-%d-%H')


        #add average volume
        
        df['avgvolumefrom'] = df['volumefrom'].rolling(window=20).mean()
        df['avgvolumeto'] = df['volumeto'].rolling(window=20).mean()
        df["avgvolumefromspike"] = np.where(df['volumefrom'] > 2*df["avgvolumefrom"], 1, 0)
        df["avgvolumetospike"] = np.where(df['volumeto'] > 2*df["avgvolumeto"], 1, 0)
        df["spikeRatio"] = (df["volumeto"]/df["avgvolumeto"])
        df["signal"] = np.where(df['spikeRatio'] > 1, 1, 0)


        pricesDfList.append(df)

    return pricesDfList
 
def highlight_signal(s):
    return ['background-color: #69420B']*len(s) if s.signal else ['background-color: #0E131E']*len(s)

# Plot raw data
def plot_raw_data(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['datetimes'], y=df['open'], name="price_open"))
    fig.add_trace(go.Scatter(x=df['datetimes'], y=df['close'], name="price_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def prophetforecast(data):
    # Predict forecast with Prophet.
    df_train = data[['datetimes','close']]
    df_train = df_train.rename(columns={"datetimes": "ds", "close": "y"})

    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    # Show and plot forecast
    #st.subheader('Forecast data')
    #st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = model.plot_components(forecast)
    st.write(fig2)



st.header("A simple Crypto Screener for  volume signals")
pricesDfListDaily=getPriceListData("daily", tickers)
pricesDfListHourly=getPriceListData("hourly", tickers)
tickerSelected = st.selectbox("Ticker", tickers)

st.subheader("Signals")

st.write("Daily")
tickerIndexD = tickers.index(tickerSelected)
figDaily=create_candlestickchart(pricesDfListDaily[tickerIndexD])
st.plotly_chart(figDaily)
#st.write(pricesDfListDaily[tickerIndexD])
st.dataframe(pricesDfListDaily[tickerIndexD].style.apply(highlight_signal, axis=1))

st.write("Hourly")
tickerIndexH = tickers.index(tickerSelected)
figHourly=create_candlestickchart(pricesDfListHourly[tickerIndexH])
st.plotly_chart(figHourly)
#st.write(pricesDfListHourly[tickerIndexH])
st.dataframe(pricesDfListHourly[tickerIndexH].style.apply(highlight_signal, axis=1))

st.subheader("Prediction")



n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

#plot_raw_data(pricesDfListHourly[tickerIndexH])
#prophetforecast(pricesDfListHourly[tickerIndexH])

#plot_raw_data(pricesDfListDaily[tickerIndexD])
prophetforecast(pricesDfListDaily[tickerIndexD])

