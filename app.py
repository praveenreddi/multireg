# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np

# # Load your saved multi-output regression model
# model = joblib.load('xgmulti.pickle_new')

# # Create the Streamlit web app
# st.title('Multi-Output Regression Predictor')

# # Create input fields for 16 variables
# input_data = []
# for i in range(16):
#    input_data.append(st.number_input(f'Input {i+1}', value=0.0))

# # Create a button to trigger the prediction
# if st.button('Predict'):
#    # Prepare the input data for the model
#    input_array = np.array(input_data).reshape(1, -1)

#    # Make predictions using the loaded model
#    predictions = model.predict(input_array)

#    # Display the results for the 6 output variables
#    for i, pred in enumerate(predictions[0]):
#        st.write(f'Output {i+1}: {pred}')

# # Add any other UI elements as needed

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your saved multi-output regression model
model = joblib.load('xgmulti.pickle_new')

# Create the Streamlit web app
st.title('Multi-Output Regression Predictor')

# Create a dictionary to map variable names to their input values
variable_names = {
   "Variable 1": st.number_input('LK_RD_', value=0.0),
   "Variable 2": st.number_input('HK_RD_', value=0.0),
   "Variable 3": st.number_input('LGO_DRYER_INLET_FLOW_', value=0.0),
   "Variable 4": st.number_input('HGO_RUNDOWN_', value=0.0),
   "Variable 5": st.number_input('HK_PA_', value=0.0),
   "Variable 6": st.number_input('LGO_PA_', value=0.0),
   "Variable 7": st.number_input('HGO_PA_', value=0.0),
   "Variable 8": st.number_input('LK_STRIPPING_STEAM_', value=0.0),
   "Variable 9": st.number_input('A-COL_STRIPPING_STEAM_', value=0.0),
   "Variable 10": st.number_input('EXP_NAPH', value=0.0),
   "Variable 11": st.number_input('EXP_KERO', value=0.0),
   "Variable 12": st.number_input('EXP_DSL', value=0.0),
   "Variable 13": st.number_input('EXP_RC', value=0.0),
   "Variable 14": st.number_input('Column_overhead_pressure', value=0.0),
   "Variable 15": st.number_input('column_top_temperature', value=0.0),
   "Variable 16": st.number_input('F-101_COMMON_OUT', value=0.0)
   
   # Add more variables as needed
}

# Create a button to trigger the prediction
if st.button('Predict'):
   # Prepare the input data for the model
   input_data = [value for value in variable_names.values()]
   input_array = np.array(input_data).reshape(1, -1)

   # Make predictions using the loaded model
   predictions = model.predict(input_array)

   # Display the results for the 6 output variables
   for i, pred in enumerate(predictions[0]):
       st.write(f'Output {i+1}: {pred}')

# Add any other UI elements as needed