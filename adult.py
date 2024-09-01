import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import urllib.request

# Download the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
urllib.request.urlretrieve(url, "adult.data")

# Load the data
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
data = pd.read_csv("adult.data", names=column_names, sep=r'\s*,\s*', engine='python', na_values="?")

# Separate features and target
X = data.drop("income", axis=1)
y = data["income"].map({">50K": 1, "<=50K": 0})

# Handle missing values
imputer = SimpleImputer(strategy="most_frequent")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Encode categorical variables
categorical_columns = X.select_dtypes(include=["object"]).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Convert to numpy arrays and save
X_np = X.to_numpy().astype(np.float32)
y_np = y.to_numpy().astype(np.int32)

np.save("adult_processed_x.npy", X_np)
np.save("adult_processed_y.npy", y_np)

print("Adult dataset preprocessed and saved as 'adult_processed_x.npy' and 'adult_processed_y.npy'")