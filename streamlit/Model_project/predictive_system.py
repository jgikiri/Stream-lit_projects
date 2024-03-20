import numpy as np
import pickle

# Loading the save model
loaded_model = pickle.load(open("D:/Desktop/New School ML/model.sav", "rb"))

# Creating a function for prediction


input_data = (200000, 300, 10000, 500000, 10, 2000000, 60, 36, 24, 10)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The school wil not Default')
else:
    print('The school will default')
