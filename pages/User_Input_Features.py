import streamlit as st 
import pandas as pd    
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler,StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer,make_column_selector

from pages import machine_learning_algorithms

reg_list,reg_list_name = machine_learning_algorithms.list_of_algorithm()

st.title('User Input Features')

def read_data():
    df = pd.read_csv("./CardioGoodFitness.csv")
    return df 

def get_unique_values(df):
    Products =  df.Product.unique()
    Genders =  df.Gender.unique()
    MaritalStatus =  df.MaritalStatus.unique()
    Usages =  df.Usage.unique()
    Fitnesss =  df.Fitness.unique()
    Education =  df.Education.unique()
    
    return Products,Genders, MaritalStatus, Usages, Fitnesss,Education

df = read_data()
Product,Genders, MaritalStatus, Usages, Fitnesss,Education = get_unique_values(df)

Education= np.sort(Education)


class CollectUserInput():
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            Products_selectbox = st.sidebar.selectbox( "Product: ",Product )
            Age_slider = st.sidebar.slider("AGE:",18,50,25)
            Genders_selectbox = st.sidebar.selectbox("Genders",Genders)
            Education_slider = st.sidebar.slider("Education:",12,21,16 )
            MaritalStatus_selectbox = st.sidebar.selectbox(  "MaritalStatus: ",MaritalStatus)    

            Usages_slider = st.sidebar.slider( "Choose your Usages level",2,7,int(4))
            Fitness_slider = st.sidebar.slider( "Choose your Fitness level",1,5,int(3) )
            income_slider = st.sidebar.slider("Choose your income level",29562,104581,59741)
            data = {'Product': Products_selectbox,
                    'Age': Age_slider,
                    'Gender': Genders_selectbox,
                    'Education': Education_slider,
                    'MaritalStatus': MaritalStatus_selectbox,
                    'Usage': Usages_slider,
                    'Fitness':Fitness_slider,
                    'Income':income_slider,

                    'Miles':0,
                    }
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()
        
        
class DataPreprocessing():
    def __init__(self, df,X):
        self.df= df
        self.X= X
    def factorialing(self):

        self.df['Product']=pd.factorize(self.df.Product)[0]
        self.df['Gender']=pd.factorize(self.df.Gender)[0]
        self.df['MaritalStatus']=pd.factorize(self.df.MaritalStatus)[0]
        return self.df

    def scalering(self):
        oh=OneHotEncoder(drop='first',handle_unknown='ignore')
        s1=StandardScaler()
        m1=MinMaxScaler()
        ct=ColumnTransformer(
                        [
                            ('cat_encoder',oh,make_column_selector(dtype_include='object')),
                            ('StandardScaler',s1,make_column_selector(dtype_include='object')),
                            ('Numerical Scaler',m1,make_column_selector(dtype_exclude='object')),  
                        ]
                        ,remainder='passthrough')
        self.X=ct.fit_transform(self.X)
        return self.X


X= pd.DataFrame()
V= pd.DataFrame()
collect_user_input=CollectUserInput()
Preprocessing=DataPreprocessing(df,X)
df1 = pd.concat([df,collect_user_input.input_df],axis=0)

X=Preprocessing.factorialing() 
Preprocessing=DataPreprocessing(V,X)

X=Preprocessing.scalering()



st.write('User input',collect_user_input.input_df)


Algorithm_selectbox = st.selectbox(
    "Select the Regressor algorithm ",reg_list_name
)

index = reg_list_name.index(Algorithm_selectbox)

load_clf = pickle.load(open(f'models/{reg_list_name[index]}.pkl', 'rb'))

prediction = load_clf.predict(X[-1:,:-1])

st.subheader('Prediction')
st.write(prediction)



# st.subheader('Prediction Probability')
# st.write(prediction_proba)
