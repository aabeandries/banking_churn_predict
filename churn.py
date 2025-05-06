import numpy as np
import pandas as pd

# import model final
from xgboost import XGBClassifier

# untuk load model
import pickle
import joblib

import streamlit as st

# -----------------

# judul
st.write("""
         <div style="text-align: center;">
         <h2>Churn Customer Prediction 🤑</h2>
         </div>
         <p>Model ini digunakan untuk memprediksi apakah seorang pelanggan akan berhenti menggunakan layanan 
         perbankan atau tetap menggunakan layanan perbankan.</p>
         <p>Model ini menggunakan algoritma XGBoost untuk melakukan prediksi berdasarkan data perilaku customer.</p>
         """, unsafe_allow_html=True)

# sidebar menu for input
st.sidebar.header("Please input your Customer's Features")

def user_input_feature():
    # untuk input numerical data
    cred_score = st.sidebar.slider(label = "Credit Score", 
                    min_value = 350, 
                    max_value = 850, value = 500)
    salary = st.sidebar.slider(label = "EstimatedSalary",
                            min_value = 11, 
                            max_value = 200000, value = 50000)
    balance = st.sidebar.slider(label = "Balance",
                            min_value = 0, 
                            max_value = 260000, value = 50000)
    age = st.sidebar.number_input(label = "Age",
                            min_value = 18, 
                            max_value = 92, value = 30)
    tenure = st.sidebar.number_input(label = "Tenure",
                            min_value = 0, 
                            max_value = 10, value = 5)
    numb_prod = st.sidebar.number_input(label = "Number of Products",
                            min_value = 1,
                            max_value = 5, value = 2)

    # untuk input categorical data

    member = st.sidebar.radio(label = "Active Member",
                        options = [0, 1], index = 1)
    cred_card = st.sidebar.radio(label = "Has Credit Card",
                        options = [0,1], index = 1)
    geo = st.sidebar.selectbox(label = "Geography",
                        options = ['France', 'Germany', 'Spain'], index = 0)
    gender = st.sidebar.radio(label='Gender',
                    options = ['Male','Female'], index = 0)

    df = pd.DataFrame()
    df['CreditScore'] = [cred_score]
    df['Geography'] = [geo]
    df['Gender'] = [gender]
    df['Age'] = [age]
    df['Tenure'] = [tenure]
    df['Balance'] = [balance]
    df['NumOfProducts'] = [numb_prod]
    df['HasCrCard'] = [cred_card]
    df['IsActiveMember'] = [member]
    df['EstimatedSalary'] = [salary]

    return df

df_feature = user_input_feature()


# load model
model = joblib.load('model_xgboost_joblib')

# predict
pred = model.predict(df_feature)

if pred[0] == 1:
    print('Customer is likely to churn')
else:
    print('Customer is predicted to stay')

left, right = st.columns(2)
with left:
    st.subheader("Customer Characteristics")
    st.write(df_feature.transpose())

with right:
    st.subheader("Prediction Result")
    if pred == [1]:
        st.write("<h3 style = 'color : red;'>😝 Your Customer is likely to CHURN 😝</h4>", unsafe_allow_html=True)
    else:
        st.write("<h3 style = 'color : green;'>😚 Your Customer is predicted to STAY 😚</h4>", unsafe_allow_html=True)
    st.write("Probability of Churn: ", model.predict_proba(df_feature)[0][1] if pred[0] == 1 else model.predict_proba(df_feature)[0][0])   
    
        