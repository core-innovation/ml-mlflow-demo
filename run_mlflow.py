import catboost as cb
import mlflow
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("income")

# Load data
dset = fetch_california_housing()
data = dset["data"]
y = dset["target"]
LABEL = dset["target_names"][0]

NUMERIC_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Longitude",
    "Latitude",
]
FEATURES = NUMERIC_FEATURES

data = pd.DataFrame(data, columns=dset["feature_names"])
data[LABEL] = y

# Split data
train_data, test_data = train_test_split(data, test_size=0.2)
print(f"Train dataset shape: {train_data.shape}")
print(f"Test dataset shape: {test_data.shape}")

X_train, X_val = train_test_split(train_data, test_size=0.2)

sc = StandardScaler()
X_train.loc[:, NUMERIC_FEATURES] = sc.fit_transform(X_train[NUMERIC_FEATURES])
X_val.loc[:, NUMERIC_FEATURES] = sc.transform(X_val[NUMERIC_FEATURES])
test_data.loc[:, NUMERIC_FEATURES] = sc.transform(test_data[NUMERIC_FEATURES])

# Start mlflow for random forest
mlflow.sklearn.autolog(disable=True)

with mlflow.start_run(run_name="rf_baseline"):

    mlflow.set_tag("model_name", "RF")

    mlflow.log_metric("n_estimators", 100)
    mlflow.log_metric("max_depth", 20)

    rf = RandomForestRegressor(n_estimators=100, max_depth=20)
    rf.fit(X_train[FEATURES], X_train[LABEL])

    rf_preds = rf.predict(test_data[FEATURES])
    rf_rms = root_mean_squared_error(test_data[LABEL], rf_preds)

    for i in range(100):
        mlflow.log_metric("test_rmse", rf_rms, step=i)

    mlflow.sklearn.log_model(rf, "sk_models")

# Start mlflow for catboost
catb_train_dataset = cb.Pool(X_train[FEATURES], X_train[LABEL])
catb_val_dataset = cb.Pool(X_val[FEATURES], X_val[LABEL])
catb_test_dataset = cb.Pool(test_data[FEATURES], test_data[LABEL])

with mlflow.start_run(run_name="catboost"):
    mlflow.set_tag("model_name", "CatBoost")
    catb = cb.CatBoostRegressor()
    catb.fit(catb_train_dataset, eval_set=catb_val_dataset, early_stopping_rounds=50)
    catb_preds = catb.predict(catb_test_dataset)
    catb_rms = root_mean_squared_error(test_data[LABEL], catb_preds)

    for i in range(100):
        mlflow.log_metric("test_rmse", catb_rms, step=i)

    mlflow.catboost.log_model(catb, "cb_models")
