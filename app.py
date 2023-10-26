# Import necessary libraries
import joblib
import streamlit as st
import pandas as pd

# Define the list of features used for training the model
features = ['Total_Stops', 'Journey_day', 'Journey_month', 'Duration_hours',
            'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
            'Airline_Jet Airways Business', 'Airline_Multiple carriers',
            'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
            'Airline_Vistara', 'Airline_Vistara Premium economy',
            'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
            'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
            'Destination_Kolkata', 'Destination_New Delhi']

# Load your pre-trained model
model_file_path = r'C:\Users\john wick\PycharmProjects\pythonProject1\knn_model.pkl'

# Load the model
model = None  # Initialize the model variable outside the try block
try:
    with open(model_file_path, 'rb') as file:
        model = joblib.load(file)
except Exception as e:
    print(f"Error: {e}")
    # Handle the exception (e.g., pr
# Replace this with your actual trained model

# Define the list of features used for training the model
features = ['Total_Stops', 'Journey_day', 'Journey_month', 'Duration_hours',
            'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
            'Airline_Jet Airways Business', 'Airline_Multiple carriers',
            'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
            'Airline_Vistara', 'Airline_Vistara Premium economy',
            'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
            'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
            'Destination_Kolkata', 'Destination_New Delhi']


# Function to predict fare based on user input
# Set the selected airline, source, and destination to 1 in the input data
def predict_fare(total_stops, journey_day, journey_month, duration_hours, airline, source, destination):
    # Prepare the input data as a DataFrame and perform prediction
    input_data = pd.DataFrame([[total_stops, journey_day, journey_month, duration_hours,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              columns=features)

    # Set the selected airline, source, and destination to 1 in the input data
    input_data['Airline_' + airline] = 1
    input_data['Source_' + source] = 1
    input_data['Destination_' + destination] = 1

    # Print input_data for debugging purposes
    print("Input Data:")
    print(input_data)

    # Predict fare using the model
    predicted_fare = model.predict(input_data)
    print("Predicted Fare:")
    print(predicted_fare)  # Print predicted_fare for debugging purposes
    return predicted_fare[0]


# Main function to handle Streamlit app
def main():
    # Streamlit app title and sidebar options
    st.title('Flight Fare Prediction')
    st.sidebar.header('Enter Flight Details')

    # User input features
    total_stops = st.sidebar.slider("Select Total Stops", 0, 2, 0)
    journey_day = st.sidebar.slider("Select Journey Day", 1, 31, 1)
    journey_month = st.sidebar.slider("Select Journey Month", 1, 12, 1)
    duration_hours = st.sidebar.slider("Select Duration Hour", 0, 24, 0)
    airline = st.sidebar.selectbox("Select Airline", ['Air India', 'GoAir', 'IndiGo',
                                                      'Jet Airways', 'Jet Airways Business',
                                                      'Multiple carriers', 'Multiple carriers Premium economy',
                                                      'SpiceJet', 'Vistara', 'Vistara Premium economy'])
    source = st.sidebar.selectbox("Select Source", ['Chennai', 'Delhi', 'Kolkata', 'Mumbai'])
    destination = st.sidebar.selectbox("Select Destination", ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi'])

    # Predict fare on user input and display the result
    if st.sidebar.button('Predict Fare'):
        predicted_fare = predict_fare(total_stops, journey_day, journey_month, duration_hours, airline, source,
                                      destination)
        st.write(f"Predicted Fare: {predicted_fare:.2f} INR")


# Run the Streamlit app
if __name__ == "__main__":
    main()
