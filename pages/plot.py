import streamlit as st 
import pandas as pd    
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_data():
    df = pd.read_csv("./CardioGoodFitness.csv")
    return df 
st.title('EDA for Cardio Good Fitness')

df= read_data()

fig = plt.figure(figsize=(8,5))
heatmat = sns.heatmap(df.select_dtypes(include="number").corr(), annot = True)
st.pyplot(fig)
corr_pairs = df.corr().unstack() 
st.write( corr_pairs[abs(corr_pairs)>0.5]) 

st.markdown('''**Observation**

* Age and Income has some in significant correlation
* Education and Income has very little correlation
* There is some corelation between Usage and Income
* Fitness and miles are corelated
* TM798 model is correlated to Education, Usage,Fitness, Income and Miles.
* Miles and usage are positively correlated''')
st.markdown("""---""")

fig = plt.figure(figsize=(8,5))
sns.regplot(data =df, x='Age',color='#C80167', y='Income',ci=75, n_boot=2000)
st.pyplot(fig)

fig = sns.relplot(
    data=df, kind="scatter",palette=['#01B0B0','#C80167','#3C2CCF'],
    x="Age", y="Income", facet_kws=dict(sharex=True),hue = 'Product',
)


st.pyplot(fig)

st.markdown('''
            **Observation**\n
            * As the age of people increases, the income also increases, as a result, the desire to use the TM798 model increases''')
st.markdown("""---""")

fig = plt.figure(figsize=(8,5))
plot = sns.countplot(x ='Product',palette=['#01B0B0','#C80167','#3C2CCF'], data = df)
plot.axes.set_title("Product Distribution",fontsize=20)
st.pyplot(fig)
 


fig = plt.figure(figsize=(8,5))
plot = sns.countplot(x ='Age',palette=['#00E4FF','#B8ED58','#6C5EF0','#B05EF0','#DE5EF0'], data = df)
plot.axes.set_title("Age Distribution", fontsize=15)
st.pyplot(fig)

st.markdown('''
            **Observation**\n

            * The use of product TM195 and TM498 is about 3.5 times that of product TM798 due to the young average age of people and their corresponding income.
            ''')
st.markdown("""---""")

g = sns.pairplot(df,hue='MaritalStatus',diag_kind="kde")
g.map_lower(sns.kdeplot, levels=4, color=".2")
st.pyplot(g)

st.markdown("""---""")
fig = plt.figure(figsize=(8,5))

sns.kdeplot(
    data=df,x="Education", y="Fitness",palette=['#01B0B0','#C80167','#3C2CCF'],hue = 'Product')
st.pyplot(fig) 
st.markdown("""---""")