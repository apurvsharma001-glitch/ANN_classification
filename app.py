import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

with open("geo_encoder.pkl","rb") as f:
    geo_encoder = pickle.load(f)


with open("label_encoder.pkl","rb") as f:
    gender_encoder = pickle.load(f)


with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

model = tf.keras.models.load_model("model.h5")

st.title("custumer_churn_prediction")

Geography = st.selectbox("Geography", geo_encoder.categories_[0])
has_cr_card = st.selectbox("credit_card", [0,1])
Gender = st.selectbox("Gender", gender_encoder.classes_)

age = st.slider("Age",18,80)
tenure = st.slider("Tenure", 1,10)
balance = st.number_input("Balance")
no_of_prod = st.number_input("Number of products")
is_active = st.selectbox("Active",[0,1])
estimated_salary = st.number_input("estimated_salary")

credit_score = st.number_input("credit_score")
geo_encodedd = geo_encoder.transform([[Geography]]).toarray()
geo_encodedd_df = pd.DataFrame(geo_encodedd, columns= geo_encoder.get_feature_names_out(["Geography"]))

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [Gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [no_of_prod],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active],
    'EstimatedSalary': [estimated_salary]
})

input_data["Gender"] = gender_encoder.transform(input_data["Gender"])
data = pd.concat((input_data,geo_encodedd_df),axis=1)

scaled = scaler.transform(data)
pred = model.predict(scaled)

if pred[0][0]>0.5:
    st.write("the customer is going to churn")
else:
    st.write("the customer is not going to churn")
