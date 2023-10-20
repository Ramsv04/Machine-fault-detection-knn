import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('D:\\SEM 3\Python for Data Science\\predictive_maintenance.csv')


X = data[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
feature_names = X.columns  # Get the feature names

y = data['Target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


st.set_page_config(page_title="Predictive Maintenance App", page_icon="⚙️")
st.markdown("<h1 style='text-align: center; color: #008000;'>Predictive Maintenance App</h1>", unsafe_allow_html=True)

def main():
    st.markdown("This app predicts whether a machine is likely to experience a failure or not based on input parameters. Please enter the following information and click the 'Predict' button:")

    st.header("Input Data")
    air_temperature = st.number_input("Air Temperature [K]")
    process_temperature = st.number_input("Process Temperature [K]")
    rotational_speed = st.number_input("Rotational Speed [rpm]")
    torque = st.number_input("Torque [Nm]")
    tool_wear = st.number_input("Tool Wear [min]")

    user_input = [air_temperature, process_temperature, rotational_speed, torque, tool_wear]

    if st.button("Predict"):
        # Make predictions
        prediction = knn.predict([user_input])

        failure_type = "Failure" if prediction[0] == 1 else "No Failure"
        if failure_type == "Failure":
            result_color = "#FF0000"
        else:
            result_color = "#008000"

        st.markdown(f"<h2 style='color: {result_color}'>Prediction Result</h2>", unsafe_allow_html=True)
        st.write(f"The predicted failure status is: {failure_type}")

if __name__ == "__main__":
    main()
