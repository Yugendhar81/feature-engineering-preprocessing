
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample dataset
data = {
    "Age": [20, 25, None, 30, 35, 28],
    "Salary": [20000, 30000, 25000, None, 50000, 40000],
    "City": ["Hyderabad", "Delhi", "Mumbai", "Hyderabad", "Delhi", "Mumbai"],
    "Purchased": [0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Separate input features and target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# Identify numerical and categorical columns
numeric_features = ["Age", "Salary"]
categorical_features = ["City"]

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Fit preprocessing on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Original dataset:")
print(df)
print("\nProcessed training data:")
print(X_train_processed)
print("\nProcessed test data:")
print(X_test_processed)
print("\nPreprocessing completed successfully!")
