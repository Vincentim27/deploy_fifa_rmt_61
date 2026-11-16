import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title='FIFA',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    st.title('FIFA Exploratory Data Analysis')

    st.header('created by: Vincent TD') #cara 1
    st.markdown('## created by: Vincent TD')#cara2

    st.image('https://png.pngtree.com/thumb_back/fh260/background/20260422/pngtree-world-cup-fifa-2026-background-image_21742119.webp',
             caption='world cup 2026')

    '''
    Menurut laporan FIFA 2022, jumlah pemain sepakbola pada tahun 2021 kurang lebih sebanyak 130.000 pemain. Namun, dalam dataset yang digunakan pada kali ini, hanya mencakup 20.000 pemain saja.

    Project kali ini bertujuan untuk memprediksi rating pemain FIFA 2022 sehingga semua pemain sepak bola profesional dapat diketahui ratingnya dan tidak menutup kemungkinan untuk lahirnya talenta/wonderkid baru.
    '''

    st.markdown('---')

    df= pd.read_csv('https://raw.githubusercontent.com/FTDS-learning-materials/phase-1/refs/heads/v2.3/w1/P1W1D1PM%20-%20Machine%20Learning%20Problem%20Framing.csv')
    st.dataframe(df)

    #first viz (barchart)
    st.header('Attacking Work Rate')
    fig = plt.figure(figsize=(16,8))
    sns.countplot(data=df,x='AttackingWorkRate')
    st.pyplot(fig)

    #second viz (histogram)
    st.header('Histogram')
    pilihan = st.selectbox('Pilih Kolom:',['Age','Height','Weight'])
    fig = plt.figure(figsize=(16,8))
    sns.histplot(df[pilihan],kde=True,bins=15)
    st.pyplot(fig)

    #third viz (scatterplot)
    st.header('Price vs Rating Correlation')
    fig= px.scatter(df,x='ValueEUR', y='Overall', hover_data=['Age','Name'])
    st.plotly_chart(fig)


if __name__ == '__main__':
    run()