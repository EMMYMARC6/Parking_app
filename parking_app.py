import streamlit as st
import pandas as pd
import joblib

# 1️⃣ Load trained model and encoders
model = joblib.load("parking_model.sav")
le_day = joblib.load("le_day.sav")
le_weather = joblib.load("le_weather.sav")

st.title("🚗 Smart Parking Prediction System")
st.write("Predict whether a parking slot is available based on Day, Hour, Weather, event and Occupied last hour.")

# 2️⃣ User inputs
day_input = st.selectbox("Select Day:", le_day.classes_)
hour_input = st.slider("Select Hour (0-23):", 0, 23, 12)
weather_input = st.selectbox("Select Weather:", le_weather.classes_)
weekend_input = st.radio("Is it Weekend?", (0, 1))
event_input = st.radio("Is there a nearby event?", (0, 1))
occupied_last_hour_input = st.radio("Was the slot occupied last hour?", (0, 1))

# 3️⃣ Encode categorical inputs
day_encoded = le_day.transform([day_input])[0]
weather_encoded = le_weather.transform([weather_input])[0]

# 4️⃣ Prepare input for prediction
sample = pd.DataFrame([[day_encoded, hour_input, weather_encoded,
                        weekend_input, event_input, occupied_last_hour_input]],
                      columns=["Day", "Hour", "Weather", "Weekend", "NearbyEvent", "OccupiedLastHour"])

# 5️⃣ Predict
if st.button("Predict Parking Availability"):
    prediction = model.predict(sample)[0]
    if prediction == 1:
        st.success("✅ You can enter, there is a free parking place.")
    else:
        st.error("❌ Sorry, no parking place available right now.")
