from flask import Flask, render_template, request
from pymongo import MongoClient
import joblib
import numpy as np
import pandas as pd
import requests
import datetime
import os
from flask_caching import Cache
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# Configure Flask-Caching
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 600})

# Load trained model and city encoder
model = joblib.load("flood_model.pkl")
city_encoder = joblib.load("city_encoder.pkl")

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")

# API URL for fetching weather data
WEATHER_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}?unitGroup=metric&key={api_key}&contentType=json"

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/flood_db")  # Default to local DB
client = MongoClient(MONGO_URI)
db = client["flood_db"]  # Database name
weather_collection = db["weather_data"]  # Collection for weather data
prediction_collection = db["flood_predictions"]  # Collection for storing predictions

def get_weather(city):
    """Fetch weather data from MongoDB first, then API if missing."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    # Check if data is already in MongoDB
    existing_data = weather_collection.find_one({"city": city, "date": today})
    if existing_data:
        print(f"Using cached weather data for {city}")
        return existing_data["weather"]

    # Fetch from API
    url = WEATHER_URL.format(location=city, api_key=API_KEY)
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "days" not in data or not data["days"]:
            return None

        current_data = data["days"][0]
        temperature = current_data.get("temp", 0)
        humidity = current_data.get("humidity", 0)
        rainfall = current_data.get("precip", current_data.get("precipitationSum", 0))

        weather_data = {"temperature": temperature, "humidity": humidity, "rainfall": rainfall}

        # Store in MongoDB for future requests
        weather_collection.insert_one({"city": city, "date": today, "weather": weather_data})

        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def delete_old_data():
    """Automatically delete predictions older than 7 days."""
    days_to_keep = 7  # Change this value if needed
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)

    # Delete records older than the cutoff date
    result = prediction_collection.delete_many({"date": {"$lt": cutoff_date.strftime("%Y-%m-%d %H:%M:%S")}})
    print(f"Deleted {result.deleted_count} old records (automated cleanup).")

# Start the background scheduler for automatic cleanup
scheduler = BackgroundScheduler()
scheduler.add_job(delete_old_data, 'interval', days=1)  # Run every 24 hours
scheduler.start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    cities = request.form['city'].split(',')
    results = []

    for city in cities:
        city = city.strip().title()  # Normalize capitalization
        weather_data = get_weather(city)

        if weather_data is None:
            results.append({
                "city": city,
                "weather": "City not found or weather data unavailable",
                "prediction": "Error"
            })
            continue

        temperature = weather_data.get("temperature", 0)
        humidity = weather_data.get("humidity", 0)
        rainfall = weather_data.get("rainfall", 0)

        print(f"City: {city} | Temp: {temperature}°C | Humidity: {humidity}% | Rainfall: {rainfall}mm")

        if city in map(str.title, city_encoder.classes_):
            city_encoded = city_encoder.transform([city])[0]
        else:
            city_encoded = -1  

        feature_names = ["Encoded City", "Temperature", "Humidity", "Rainfall"]
        input_data = pd.DataFrame([[city_encoded, temperature, humidity, rainfall]], columns=feature_names)

        try:
            predicted_risk = model.predict(input_data)[0]
            risk_labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
            risk_label = risk_labels.get(predicted_risk, "Unknown")
            print(f"Prediction for {city}: {risk_label}")
        except ValueError as e:
            print(f"Model error: {e}")
            risk_label = "Prediction Error"

        prediction_record = {
            "city": city,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "predicted_risk": risk_label
        }
        prediction_collection.insert_one(prediction_record)  # Save to MongoDB

        results.append({
            "city": city,
            "weather": f"Temp: {temperature}°C, Humidity: {humidity}%, Rainfall: {rainfall}mm",
            "prediction": risk_label
        })

    return render_template('result.html', results=results, api_key=API_KEY)

@app.route('/history', methods=['GET'])
def history():
    """View past predictions with search filters."""
    city = request.args.get("city", "").strip().title()
    date = request.args.get("date", "").strip()

    # Build search query
    query = {}
    if city:
        query["city"] = city
    if date:
        query["date"] = {"$regex": f"^{date}"}  # Search for matching date prefix

    past_predictions = list(prediction_collection.find(query, {"_id": 0}))
    
    return render_template('history.html', predictions=past_predictions)

@app.route('/delete_old_data')
def manual_delete_old_data():
    """Manually delete predictions older than 7 days."""
    days_to_keep = 7
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)

    result = prediction_collection.delete_many({"date": {"$lt": cutoff_date.strftime("%Y-%m-%d %H:%M:%S")}})
    
    return f"Deleted {result.deleted_count} old records."

if __name__ == '__main__':
    app.run(debug=True)
