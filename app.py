from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# import unicodedata
import numpy as np


app = Flask(__name__)
model = pickle.load(open("flight_xgb.pkl", "rb"))
model_rf = pickle.load(open("flight_randForest.pkl", "rb"))

# Load the label encoders
airline_encoder = LabelEncoder()
flight_class_encoder = LabelEncoder()
source_city_encoder = LabelEncoder()
destination_city_encoder = LabelEncoder()
dept_time_encoder = LabelEncoder()
arr_time_encoder = LabelEncoder()

# Load encodings for airlines
airline_encoder.fit(['Air India', 'AirAsia', 'AkasaAir', 'AllianceAir', 'GO FIRST', 'Indigo', 'SpiceJet', 'StarAir', 'Vistara'])

# Load encodings for flight classes
flight_class_encoder.fit(['BUSINESS', 'ECONOMY', 'FIRST', 'PREMIUMECONOMY'])

# Load encodings for source cities
source_city_encoder.fit(['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])

# Load encodings for destination cities
destination_city_encoder.fit(['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])

dept_time_encoder.fit(['00:05', '00:15', '00:30', '01:05', '01:10', '01:15', '01:40', '01:45', '01:55', '02:00', '02:10', '02:30', '02:40', '02:45', '03:00', '03:25', '03:30', '03:55', '04:00', '04:05', '04:10', '04:15', '04:20', '04:25', '04:30', '04:35', '04:40', '04:45', '04:50', '04:55', '05:00', '05:05', '05:10', '05:15', '05:20', '05:25', '05:30', '05:35', '05:40', '05:45', '05:50', '05:55', '06:00', '06:05', '06:10', '06:15', '06:20', '06:25', '06:30', '06:35', '06:40', '06:45', '06:50', '06:55', '07:00', '07:05', '07:10', '07:15', '07:20', '07:25', '07:30', '07:35', '07:40', '07:45', '07:50', '07:55', '08:00', '08:05', '08:10', '08:15', '08:20', '08:25', '08:30', '08:35', '08:40', '08:45', '08:50', '08:55', '09:00', '09:05', '09:10', '09:15', '09:20', '09:25', '09:30', '09:35', '09:40', '09:45', '09:50', '09:55', '10:00', '10:05', '10:10', '10:15', '10:20', '10:25', '10:30', '10:35', '10:40', '10:45', '10:50', '10:55', '11:00', '11:05', '11:10', '11:15', '11:20', '11:25', '11:30', '11:35', '11:40', '11:45', '11:50', '11:55', '12:00', '12:05', '12:10', '12:15', '12:20', '12:25', '12:30', '12:35', '12:40', '12:45', '12:50', '12:55', '13:00', '13:05', '13:10', '13:15', '13:20', '13:25', '13:30', '13:35', '13:40', '13:45', '13:50', '13:55', '14:00', '14:05', '14:10', '14:15', '14:20', '14:25', '14:30', '14:35', '14:40', '14:45', '14:50', '14:55', '15:00', '15:05', '15:10', '15:15', '15:20', '15:25', '15:30', '15:35', '15:40', '15:45', '15:50', '15:55', '16:00', '16:05', '16:10', '16:15', '16:20', '16:25', '16:30', '16:35', '16:40', '16:45', '16:50', '16:55', '17:00', '17:05', '17:10', '17:15', '17:20', '17:25', '17:30', '17:35', '17:40', '17:45', '17:50', '17:55', '18:00', '18:05', '18:10', '18:15', '18:20', '18:25', '18:30', '18:35', '18:40', '18:45', '18:50', '18:55', '19:00', '19:05', '19:10', '19:15', '19:20', '19:25', '19:30', '19:35', '19:40', '19:45', '19:50', '19:55', '20:00', '20:05', '20:10', '20:15', '20:20', '20:25', '20:30', '20:35', '20:40', '20:45', '20:50', '20:55', '21:00', '21:05', '21:10', '21:15', '21:20', '21:25', '21:30', '21:35', '21:40', '21:45', '21:50', '21:55', '22:00', '22:05', '22:10', '22:15', '22:20', '22:25', '22:30', '22:35', '22:40', '22:45', '22:50', '22:55', '23:00', '23:05', '23:10', '23:15', '23:20', '23:25', '23:30', '23:40', '23:45', '23:50', '23:55'])
arr_time_encoder.fit(['00:00', '00:05', '00:10', '00:15', '00:20', '00:25', '00:30',
       '00:35', '00:40', '00:45', '00:50', '00:55', '01:00', '01:05',
       '01:10', '01:15', '01:20', '01:25', '01:30', '01:35', '01:40',
       '01:45', '01:50', '01:55', '02:00', '02:05', '02:10', '02:15',
       '02:20', '02:30', '02:40', '02:45', '02:55', '03:00', '03:05',
       '03:45', '03:50', '03:55', '04:00', '04:15', '04:20', '04:45',
       '04:50', '05:10', '05:15', '05:25', '05:35', '05:45', '05:50',
       '05:55', '06:05', '06:10', '06:15', '06:20', '06:25', '06:30',
       '06:35', '06:40', '06:45', '06:50', '06:55', '07:00', '07:05',
       '07:10', '07:15', '07:20', '07:25', '07:30', '07:35', '07:40',
       '07:45', '07:50', '07:55', '08:00', '08:05', '08:10', '08:15',
       '08:20', '08:25', '08:30', '08:35', '08:40', '08:45', '08:50',
       '08:55', '09:00', '09:05', '09:10', '09:15', '09:20', '09:25',
       '09:30', '09:35', '09:40', '09:45', '09:50', '09:55', '10:00',
       '10:05', '10:10', '10:15', '10:20', '10:25', '10:30', '10:35',
       '10:40', '10:45', '10:50', '10:55', '11:00', '11:05', '11:10',
       '11:15', '11:20', '11:25', '11:30', '11:35', '11:40', '11:45',
       '11:50', '11:55', '12:00', '12:05', '12:10', '12:15', '12:20',
       '12:25', '12:30', '12:35', '12:40', '12:45', '12:50', '12:55',
       '13:00', '13:05', '13:10', '13:15', '13:20', '13:25', '13:30',
       '13:35', '13:40', '13:45', '13:50', '13:55', '14:00', '14:05',
       '14:10', '14:15', '14:20', '14:25', '14:30', '14:35', '14:40',
       '14:45', '14:50', '14:55', '15:00', '15:05', '15:10', '15:15',
       '15:20', '15:25', '15:30', '15:35', '15:40', '15:45', '15:50',
       '15:55', '16:00', '16:05', '16:10', '16:15', '16:20', '16:25',
       '16:30', '16:35', '16:40', '16:45', '16:50', '16:55', '17:00',
       '17:05', '17:10', '17:15', '17:20', '17:25', '17:30', '17:35',
       '17:40', '17:45', '17:50', '17:55', '18:00', '18:05', '18:10',
       '18:15', '18:20', '18:25', '18:30', '18:35', '18:40', '18:45',
       '18:50', '18:55', '19:00', '19:05', '19:10', '19:15', '19:20',
       '19:25', '19:30', '19:35', '19:40', '19:45', '19:50', '19:55',
       '20:00', '20:05', '20:10', '20:15', '20:20', '20:25', '20:30',
       '20:35', '20:40', '20:45', '20:50', '20:55', '21:00', '21:05',
       '21:10', '21:15', '21:20', '21:25', '21:30', '21:35', '21:40',
       '21:45', '21:50', '21:55', '22:00', '22:05', '22:10', '22:15',
       '22:20', '22:25', '22:30', '22:35', '22:40', '22:45', '22:50',
       '22:55', '23:00', '23:05', '23:10', '23:15', '23:20', '23:25',
       '23:30', '23:35', '23:40', '23:45', '23:50', '23:55', '23:59'])

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html") #this render the home page. 

@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # Total Stops
        total_stops = int(request.form["Total Stops"])

        # Journey Day
        journey_day = int(request.form["Journey day"])

        # Journey Month
        journey_month = int(request.form["Journey month"])

        # Airline
        airline = request.form['Airline']
        airline_encoded = airline_encoder.transform([airline])[0]

        # print(airline_encoded)


        # Flight Class
        flight_class = request.form['class']
        flight_class_encoded = flight_class_encoder.transform([flight_class])[0]

        # Departure Time
        departure_time = str(request.form["Departure time"])
        departure_time_encoded = dept_time_encoder.transform([departure_time])[0]

        # Source City
        source_city = request.form["Source city"]
        source_city_encoded = source_city_encoder.transform([source_city])[0]

        # Arrival Time
        arrival_time = str(request.form["Arrival time"])
        arrival_time_encoded = arr_time_encoder.transform([arrival_time])[0]


        # Destination City
        destination_city = request.form["Destination_city"]
        destination_city_encoded = destination_city_encoder.transform([destination_city])[0]

        #Duration
        duration_time = arrival_time_encoded-departure_time_encoded


        # Function to encode string using ASCII for xgboost
        # def encode_ascii(text):
        #     return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

        # # Encoding airline, source city, and destination city using ASCII
        # airline_adjusted = encode_ascii(airline.strip()) 
        # source_city_adjusted = encode_ascii(source_city)
        # destination_city_adjusted = encode_ascii(destination_city)

        # # Transform encoded strings
        # airline_encoded = airline_encoder.transform([airline_adjusted])[0]
        # source_city_encoded = source_city_encoder.transform([source_city_adjusted])[0]
        # destination_city_encoded = destination_city_encoder.transform([destination_city_adjusted])[0]


        '''duration 

        # Departure Time
        departure_time = int(request.form["departure_time"])

        # Arrival Time
        arrival_time = int(request.form["arrival_time"])

        # Calculate travel duration
        duration_hours = (arrival_time - departure_time) // 60
        duration_minutes = (arrival_time - departure_time) % 60

        #duration=arrival_time-departure_time'''

        to_predict = np.array([[
            duration_time,
            total_stops,
            journey_day,
            journey_month,
            airline_encoded,
            flight_class_encoded,
            departure_time_encoded,
            source_city_encoded,
            arrival_time_encoded,
            destination_city_encoded
        ]])

        # prediction = model.predict([
        #     duration_time,
        #     total_stops,
        #     journey_day,
        #     journey_month,
        #     airline_encoded,
        #     flight_class_encoded,
        #     departure_time,
        #     source_city_encoded,
        #     arrival_time,
        #     destination_city_encoded
        # ])
        

        # to_predict = np.array([[2.05, 2, 14, 8, 5, 1, 210, 3, 310, 2]])
        prediction = model.predict(to_predict)


        output = round(prediction[0], 2)

        return render_template('home.html', prediction_text="Your Flight price is Rs. {}".format(output))

    return render_template("home.html")

if __name__ == "__main__": #run the application 
    app.run(debug=True)
