import os
import joblib
from geopy.distance import geodesic
import pandas as pd
import numpy as np

def preprocess(data: dict) -> list:
    """ This Function preprocess the data before scoring the model """

    Delivery_person_ID=data['Delivery_person_ID']
    age=int(data['age'])
    rating=float(data['rating'])
    restaurant_latitude=float(data['restaurant_latitude'])
    restaurant_longitude=float(data['restaurant_longitude'])
    delivery_location_latitude=float(data['delivery_location_latitude'])
    delivery_location_longitude=float(data['delivery_location_longitude'])
    
    Order_Date=data['Order_Date']
    Order_Date=pd.to_datetime(Order_Date,format="%Y-%m-%d")
    day=Order_Date.day
    month=Order_Date.month
    year=Order_Date.year
    
    Time_Ordered=data['Time_Ordered'] + ":00"
    Time_Ordered=pd.to_timedelta(Time_Ordered)

    Time_Order_picked=data['Time_Order_picked'] + ":00"
    Time_Order_picked=pd.to_timedelta(Time_Order_picked)

    Time_Order_picked_formatted=Order_Date + np.where(Time_Order_picked < Time_Ordered, pd.DateOffset(days=1), pd.DateOffset(days=0)) + Time_Order_picked
    Time_Ordered_formatted = Order_Date + Time_Ordered
    Time_Order_picked_formatted=pd.to_datetime(Time_Order_picked_formatted)

    order_prepare_time = (Time_Order_picked_formatted - Time_Ordered_formatted).total_seconds() / 60

    distance=geodesic((restaurant_latitude,restaurant_longitude),
                (delivery_location_latitude,delivery_location_longitude)).km

    Vehicle_condition=int(data['Vehicle_condition'])
    multiple_deliveries=int(data['multiple_deliveries'])


    #load label encoders
    City_code_encoder=joblib.load(os.path.join(os.getcwd(),"model_store","City_code_encoder.joblib"))
    City_encoder=joblib.load(os.path.join(os.getcwd(),"model_store","City_encoder.joblib"))
    Festival_encoder=joblib.load(os.path.join(os.getcwd(),"model_store","Festival_encoder.joblib"))
    Road_traffic_density_encoder=joblib.load(os.path.join(os.getcwd(),"model_store","Road_traffic_density_encoder.joblib"))
    Type_of_order_encoder=joblib.load(os.path.join(os.getcwd(),"model_store","Type_of_order_encoder.joblib"))
    Type_of_vehicle_encoder=joblib.load(os.path.join(os.getcwd(),"model_store","Type_of_vehicle_encoder.joblib"))
    Weather_conditions_encoder=joblib.load(os.path.join(os.getcwd(),"model_store","Weather_conditions_encoder.joblib"))


    Weather_conditions=data['Weather_conditions']
    Weather_conditions=Weather_conditions_encoder.transform([Weather_conditions])[0]

    Road_traffic_density=data['Road_traffic_density']
    Road_traffic_density=Road_traffic_density_encoder.transform([Road_traffic_density])[0]

    order=data['order']
    order=Type_of_order_encoder.transform([order])[0]

    vehicle=data['vehicle']
    vehicle=Type_of_vehicle_encoder.transform([vehicle])[0]

    Festival=data['Festival']
    Festival=Festival_encoder.transform([Festival])[0]

    City=data['City']
    City=City_encoder.transform([City])[0]

    City_code=Delivery_person_ID.split("RES")[0]
    City_code=City_code_encoder.transform([City_code])[0]

    prediction_data=[age,rating,restaurant_latitude,restaurant_longitude,delivery_location_latitude,
                        delivery_location_longitude,Weather_conditions,Road_traffic_density,
                        Vehicle_condition,order,vehicle,multiple_deliveries,Festival,City,
                        City_code,day,month,year,order_prepare_time,distance]

    return prediction_data

def score(data: list) -> int:
    """ This Function score the model """
    path = os.path.join(os.getcwd(), "model_store", "XGBR_model.joblib")
    # load model
    model=joblib.load(path)

    # make prediction
    response = model.predict([data])
    return response[0]
