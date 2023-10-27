import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a dictionary with sample data
data = {
    'weather_condition': ['Clear', 'Rain', 'Snow', 'Fog', 'Rain'],
    'road_type': ['Highway', 'City Street', 'Rural Road', 'City Street', 'Highway'],
    'speed_limit': [65, 30, 55, 30, 70],
    'vehicle_type': ['Car', 'Truck', 'Motorcycle', 'Car', 'Car'],
    'accident_severity': ['Minor', 'Major', 'Minor', 'Major', 'Minor']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
df.to_csv('accident_data.csv', index=False)

# Check the created dataset
print(df)

# Load the dataset
data = pd.read_csv('accident_data.csv')

# Explore the dataset
data.head()

# Encode categorical variables
data = pd.get_dummies(data, columns=['weather_condition', 'road_type', 'vehicle_type'], drop_first=True)

# Define independent and dependent variables
X = data.drop('accident_severity', axis=1)
y = data['accident_severity']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

model_filename = "accident_severity_model.pkl"
joblib.dump(model, model_filename)

# Load the saved model
loaded_model = joblib.load(model_filename)

# Create a sample input
sample_input = pd.DataFrame({
    'weather_condition_Clear': [1],
    'road_type_Rural': [0],
    'speed_limit': [40],
    'vehicle_type_Car': [1]
})

# Make a prediction
predicted_severity = loaded_model.predict(sample_input)
print("Predicted Accident Severity:", predicted_severity[0])
