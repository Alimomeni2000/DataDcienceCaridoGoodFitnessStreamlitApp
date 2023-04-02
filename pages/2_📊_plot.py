import streamlit as st 
import pandas as pd    
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import random

class Plot():
    def __init__(self,df):
        self.palette= ['#3A666D','#F3FA0C','#ECE506','#18121B','#5E2828','#00E4FF','#B8ED58','#04AD04','#9BAA00','#F20000','#78587C','#3855D8','#282B5E','#DE5EF0']
        self.df= df
        st.title('EDA for Cardio Good Fitness')

    def histogram_plot(self):
        x_axis_val = st.selectbox('Select X_Axis Vlaue', options= self.df.columns)
        y_axis_val = st.selectbox('Select Y_Axis Vlaue', options= self.df.columns)
        fig = px.histogram(self.df, x=x_axis_val,color=y_axis_val)
        fig.update_layout(bargap=0.45)
        st.plotly_chart(fig)
        
    def heatmap(self):
        fig =px.imshow(self.df.select_dtypes(include="number").corr(),text_auto=True,aspect="auto" )
        st.plotly_chart(fig)

        corr_pairs = self.df.corr().unstack() 
        st.write( corr_pairs[abs(corr_pairs)>0.5]) 

        st.markdown('''
            **Observation**
            * Age and Income has some in significant correlation
            * Education and Income has very little correlation
            * There is some corelation between Usage and Income
            * Fitness and miles are corelated
            * TM798 model is correlated to Education, Usage,Fitness, Income and Miles.
            * Miles and usage are positively correlated
            ''')


    def regplot(self,):
        columns = ['Age', 'Education', 'Usage','Fitness','Income','Miles','Product']
        x_axis_val = st.selectbox('Select X_Axis Vlaue', options= columns)
        y_axis_val = st.selectbox('Select Y_Axis Vlaue', options= columns)
        fig = px.scatter( self.df, x=x_axis_val, y=y_axis_val,
        trendline='ols', trendline_color_override='darkblue' )
        st.plotly_chart(fig)

    def scatter(self):
        columns = ['Age', 'Education', 'Usage','Fitness','Income','Miles']
        hue_columns = ['Gender', 'Product','Fitness','Usage','Education','MaritalStatus']
        x_axis_val = st.selectbox('Select X_Axis Vlaue', options= columns)
        y_axis_val = st.selectbox('Select Y_Axis Vlaue', options= columns)
        hue = st.selectbox('Select Hue Vlaue', options= hue_columns)
        random.shuffle(self.palette)
        self.palette=self.palette[:len(self.df[hue].unique())]
        fig = sns.relplot(data=self.df, kind="scatter",
        x=x_axis_val, y=y_axis_val, palette=self.palette,
        facet_kws=dict(sharex=True),hue = hue, )

        st.pyplot(fig)


def read_data():
    df = pd.read_csv("./CardioGoodFitness.csv")
    return df 
     
df= read_data()        
plot =Plot(df)
st.subheader('Select a chart 👇')
plots_selectbox = st.selectbox('',['Heatmap', 'Histogram','Regplot','Scatter'])
st.markdown("""---""")  

if plots_selectbox == 'Heatmap':
    plot.heatmap()
    st.markdown("""---""")  
    
elif plots_selectbox =='Histogram':
    plot.histogram_plot()
    st.markdown("""---""")
    
elif plots_selectbox =='Regplot': 
    plot.regplot()
    st.markdown("""---""")
    
elif plots_selectbox =='Scatter':
    plot.scatter()
