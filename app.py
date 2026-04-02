import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AQI Dashboard", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0F172A;
}
.card {
    background-color: #1E293B;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    color: white;
}
.metric {
    font-size: 26px;
    font-weight: bold;
}
.label {
    color: #94A3B8;
}
</style>
""", unsafe_allow_html=True)

data = pd.read_csv("city_day.csv")

if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])

data = data.dropna(subset=['AQI'])

pollutants = ['PM2.5','PM10','NO2','SO2','CO','O3']
data[pollutants] = data[pollutants].fillna(data[pollutants].median())

model = pickle.load(open("model.pkl", "rb"))

city = st.sidebar.selectbox("Select City", data['City'].unique())
city_data = data[data['City'] == city].sort_values(by='Date')

if city_data.empty:
    st.error("No data available for selected city")
    st.stop()

latest = city_data.iloc[-1]

latest_aqi = latest['AQI']
latest_aqi = int(latest_aqi) if pd.notna(latest_aqi) else 0

avg_aqi = round(city_data['AQI'].mean(), 2)

st.title("🌍 Air Quality Index Prediction Dashboard")

c1, c2, c3 = st.columns(3)

c1.markdown(f"""
<div class="card">
<div class="metric">{city}</div>
<div class="label">City</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="card">
<div class="metric">{latest_aqi}</div>
<div class="label">Latest AQI</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="card">
<div class="metric">{avg_aqi}</div>
<div class="label">Average AQI</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1,1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Enter Pollutant Values")

    pm25 = st.number_input("PM2.5", value=float(latest['PM2.5']))
    pm10 = st.number_input("PM10", value=float(latest['PM10']))
    no2 = st.number_input("NO2", value=float(latest['NO2']))
    so2 = st.number_input("SO2", value=float(latest['SO2']))
    co = st.number_input("CO", value=float(latest['CO']))
    o3 = st.number_input("O3", value=float(latest['O3']))

    if st.button("🔍 Predict AQI"):

        input_data = pd.DataFrame([[pm25, pm10, no2, so2, co, o3]],
                                  columns=pollutants)

        prediction = model.predict(input_data)[0]
        aqi = int(prediction)

        # AQI Category
        if aqi <= 50:
            color = "#00E400"
            category = "Good"
            advice = "Air quality is satisfactory."
        elif aqi <= 100:
            color = "#FFFF00"
            category = "Moderate"
            advice = "Sensitive people should take care."
        elif aqi <= 200:
            color = "#FF7E00"
            category = "Poor"
            advice = "Avoid outdoor activities."
        elif aqi <= 300:
            color = "#FF0000"
            category = "Very Poor"
            advice = "Limit outdoor exposure."
        else:
            color = "#8F3F97"
            category = "Severe"
            advice = "Stay indoors."

        st.markdown(f"""
        <div style="
            background:{color};
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:20px;
            color:black;">
            <b>AQI: {aqi}</b><br>
            {category}<br>
            {advice}
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(aqi/500,1.0))

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 What affects AQI most?")

    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(pollutants, importances)

    st.pyplot(fig)

    st.info("Higher value = more influence on AQI prediction")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 AQI Trend Over Time")

if not city_data['AQI'].isna().all():
    st.line_chart(city_data.set_index('Date')['AQI'])
else:
    st.warning("No AQI trend data available")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:gray; margin-top:20px;">
AQI Prediction System • Machine Learning Dashboard • © 2026
</div>
""", unsafe_allow_html=True)