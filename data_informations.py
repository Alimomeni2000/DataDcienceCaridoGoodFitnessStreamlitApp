import streamlit as st

st.set_page_config(
    page_title="Cardio Good Fitness App",
    page_icon="👋",
)
st.title('Cardio Good Fitness App')

st.markdown('''
         
* The market research team at AdRight is assigned the task to identify the profile of the typical customer for each treadmill product offered by CardioGood Fitness.\n

* The market research team decides to investigate whether there are differences across the product lines with respect to customer characteristics.\n

* The team decides to collect data on individuals who purchased a treadmill at a CardioGoodFitness retail store during the prior three months.\n

* The data are stored in the CardioGoodFitness.csv file\n\n\n

**The team identifies the following customer variables to study:**\n
1- product purchased, TM195, TM498, or TM798\n
2- gender;\n
3- age, in years;\n
4- education, in years;\n
5- relationship status, single or partnered;\n
6- annual household income ;\n
7- average number of times the customer plans to use the treadmill each week;\n
8- average number of miles the customer expects to walk/run each week;\n
9- and self-rated fitness on an 1-to-5 scale, where 1 is poor shape and 5 is excellent shape.\n

you can download data in this link:
https://www.kaggle.com/datasets/saurav9786/cardiogoodfitness
         ''')