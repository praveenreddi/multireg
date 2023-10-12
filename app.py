import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your saved multi-output regression model
model = joblib.load('xgmulti.pickle_new')

# Create the Streamlit web app
st.title('Multi-Output Regression Predictor')

# Create input fields for 16 variables
input_data = []
for i in range(16):
   input_data.append(st.number_input(f'Input {i+1}', value=0.0))

# Create a button to trigger the prediction
if st.button('Predict'):
   # Prepare the input data for the model
   input_array = np.array(input_data).reshape(1, -1)

   # Make predictions using the loaded model
   predictions = model.predict(input_array)

   # Display the results for the 6 output variables
   for i, pred in enumerate(predictions[0]):
       st.write(f'Output {i+1}: {pred}')

# Add any other UI elements as needed