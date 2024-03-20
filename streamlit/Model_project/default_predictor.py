import numpy as np
import pickle
import streamlit as st
import os

# Loading the save model
model_path = os.path.join("D:", os.sep, "Desktop", "New School ML", "model.sav")
loaded_model = pickle.load(open(model_path, "rb"))

# Creating a function for prediction


def default_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The school wil not Default'
    else:
        return 'The school will default'


def main():
    # Title
    st.title('Default Predictor Web App')

    # Getting input data from the user
    Outstanding_Loan = st.number_input('Outstanding Loan')
    Student_Count = st.number_input('Student Count')
    Annual_Fee = st.number_input('Monthly Fee')
    AnnualFee_Income = st.number_input('Annual Fee')
    Teacher_Count = st.number_input('Teacher Count')
    Net_Disbursed = st.number_input('Net Disbursement')
    Tenure = st.number_input('Tenure')
    Interest_Rate = st.number_input('Interest Rate')
    Vintage = st.number_input('Vintage')
    ST_Ratio = st.number_input('Student Teacher Ratio')

    # Prediction code
    prediction = ''

    # Creating a button of default
    if st.button('Default Result'):
        prediction = default_prediction([Outstanding_Loan, Student_Count, Annual_Fee,
                                         AnnualFee_Income, Teacher_Count, Net_Disbursed, Tenure, Interest_Rate,
                                         Vintage, ST_Ratio])
    st.success(prediction)

if __name__ == "__main__":
    main()