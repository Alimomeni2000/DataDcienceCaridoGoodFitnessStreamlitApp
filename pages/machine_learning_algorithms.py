import streamlit as st 
import pandas as pd    
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler,StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn import  preprocessing



def read_data():
    df = pd.read_csv("./CardioGoodFitness.csv")
    return df 
st.title('Regression algorithms')

def factorialing(df):
    df['Product']=pd.factorize(df.Product)[0]
    df['Gender']=pd.factorize(df.Gender)[0]
    df['MaritalStatus']=pd.factorize(df.MaritalStatus)[0]
    return df

def scalering(X):
    oh=OneHotEncoder(drop='first',handle_unknown='ignore')
    s1=StandardScaler()
    m1=MinMaxScaler()

    ct=ColumnTransformer(
                    [
                        ('cat_encoder',oh,make_column_selector(dtype_include='object')),
                        ('StandardSinput_dfcaler',s1,make_column_selector(dtype_include='object')),
                        ('Numerical Scaler',m1,make_column_selector(dtype_exclude='object')),  
                    ]
                    ,remainder='passthrough')
    X=ct.fit_transform(X)
    return X

def list_of_algorithm():

    lr = LinearRegression(n_jobs=-1)
    svr = SVR(max_iter=2000)
    dtr = DecisionTreeRegressor()
    knnr = KNeighborsRegressor(n_neighbors=3,n_jobs=-1)
    rfr = RandomForestRegressor(n_jobs=-1,n_estimators= 900,)
    abr = AdaBoostRegressor(n_estimators=2000,learning_rate=0.85)
    cbr = CatBoostRegressor(iterations=400, learning_rate=.80,depth=16,eval_metric='MAE', verbose=200)
    xg_reg  =xgb.XGBRFRegressor(colsample_bynode=1,max_depth=35,min_child_weight= 11,alpha= 6.9353429991712695e-08,
        subsample= 1,colsample_bytree= 0.99,gamma= 1,
                            n_estimators=300,n_jobs=-1,random_state=23,grow_policy= 'depthwise')
    reg_list=[lr,svr,dtr,knnr,rfr,abr,xg_reg]

    reg_list_name = ['LinearRegression','SVR','DecisionTreeRegressor','KNeighborsRegressor',
             'RandomForestRegressor','AdaBoostRegressor','XGBRFRegressor']
    
    return reg_list,reg_list_name



def fitting_predict_al(reg, X,y):
    
    start = time.time()

    reg.fit(X_train,y_train.values.ravel())
 
    reg_name = reg.__class__.__name__
    
    pred=reg.predict(X_test)
    
    mse = mean_squared_error(pred, y_test)
    rmse = np.sqrt(mse)
    st.write(reg_name)
    score_dict =[]
    score_dict.append({
        "R2 Score":r2_score(y_test,pred),
        "Mean Absolute Error Score":mean_absolute_error(y_test,pred),
        "Mean Squared Error Score":mean_squared_error(y_test,pred),
        "RMSE" : rmse,
    })
    
    st.write(pd.DataFrame(score_dict))
    st.write(f"R2 Score : {r2_score(y_test,pred)}")
    st.write(f"Mean Absolute Error Score : {mean_absolute_error(y_test,pred)}")
    st.write(f"Mean Squared Error Score : {mean_squared_error(y_test,pred)}")
    st.write(f"RMSE : {rmse}")
    end = time.time()
    st.markdown(f'**Total learning and prediction time of {reg_name} Algorithm is:** {round(end - start,3)} seconds') 

    
df= read_data()
X = df.drop(['Miles'],axis=1)
y= df['Miles']   
X=factorialing(X)   

X=scalering(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=101)

reg_list,reg_list_name = list_of_algorithm()

Algorithm_selectbox = st.selectbox(
    "Select the Regressor algorithm ",reg_list_name
)

index = reg_list_name.index(Algorithm_selectbox)
st.write(reg_list[index])

fitting_predict_al(reg_list[index],X,y) 
