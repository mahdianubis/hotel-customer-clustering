import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("hotel-customer-clustering/data/raw/hotel_bookings.csv")
df = df.drop(["company", "arrival_date_week_number", "reservation_status", "reservation_status_date", "arrival_date_year", "agent", "arrival_date_day_of_month"], axis=1)
df = df.dropna()

le = LabelEncoder()

top_10_country = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD"]
df["country"] = df["country"].apply(lambda x: x if x in top_10_country else "Other")

lst = ["country", "hotel", "customer_type", "meal", "market_segment", "distribution_channel", "deposit_type"]

for x in lst:
    le = LabelEncoder()
    df[x] = le.fit_transform(df[x])

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
df["arrival_date_month"] = df["arrival_date_month"].apply(lambda x: months.index(x) + 1)

df["is_room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
room_ranking = {
    "B": 1, "A": 1, "D": 2, "E": 3,
    "L": 3, "C": 4, "F": 5, "G": 6,
    "H": 7,"P": 0  
}
df["reserved_room_type"] = df["reserved_room_type"].map(room_ranking)

df["total_stays_in_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

df = df.drop(["assigned_room_type", "stays_in_week_nights", "stays_in_weekend_nights"], axis=1)

df.to_csv("hotel-customer-clustering/data/processed/Cleared_data.csv", index=False)
