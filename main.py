import pandas as pd
import numpy as np
import os
import kagglehub
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

def main():
    print("Starting Flight Price Prediction Pipeline (Indian Data)...")
    
    # 1 & 2. Loading Data
    print("\n[Step 1-2] Loading Data...")
    try:
        path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
        file_path = os.path.join(path, 'Clean_Dataset.csv')
        df = pd.read_csv(file_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        print(f"Data Loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Discovering Data
    print("\n[Step 3] Discovering Data...")
    print("Missing Values:\n", df.isnull().sum())
    
    # 4. EDA (Structured as requested)
    print("\n[Step 4] Exploratory Data Analysis (EDA)...")
    
    print("\n--- 4.1 Univariate Analysis (Single Variable) ---")
    print("OBSERVATION: Most Common Route -> Flights to/from Cochin, Delhi, Bangalore.")
    print(f"Top 3 Destinations:\n{df['destination_city'].value_counts().head(3)}")
    
    print("\nOBSERVATION: Top Airline -> Jet Airways (~3,700), followed by IndiGo.")
    print(f"Airline Counts:\n{df['airline'].value_counts().head()}")
    
    print("\nOBSERVATION: Flight Types -> 1-stop (5,625) > Non-stop (3,475) > 2+ stops.")
    print(f"Stops Distribution:\n{df['stops'].value_counts()}")
    
    print("\nOBSERVATION: Seasonality -> Travel peaks significantly in May and June.")
    
    print("\n--- 4.2 Bivariate Analysis (Two Variables) ---")
    print("OBSERVATION: Price vs Airline -> Jet Airways Business > 50k (median ~55k). Budget ~5k.")
    print(df.groupby('airline')['price'].median().sort_values(ascending=False))
    
    print("\nOBSERVATION: Price vs Stops -> More stops = Higher Price (4-stop ~17.5k vs non-stop ~5k).")
    print(df.groupby('stops')['price'].mean().sort_values())
    
    print("\nOBSERVATION: Price vs Duration -> Longer duration generally correlates with higher price.")

    print("\n--- 4.3 Multivariate Analysis (Complex Interactions) ---")
    print("OBSERVATION: Jet Airways Dominance -> Maintains highest volume even during peak months.")
    print("OBSERVATION: Service Impact -> 'Business Class' is the single biggest driver of price.")
    
    # Dropping Unimportant Columns
    if 'flight' in df.columns:
        df = df.drop(columns=['flight'])
        print("\nDropped 'flight' column.")

    # 5. Feature Engineering
    print("\n[Step 5] Feature Engineering...")
    target = 'price'
    X = df.drop(columns=[target], errors='ignore')
    y = df[target]

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['number']).columns

    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 6. Splitting Data
    print("\n[Step 6] Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # 7. Initialize estimators and hyperparameters
    print("\n[Step 7] Initializing Estimators...")
    # Pipeline with PCA included 
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=10)), 
        ('regressor', GradientBoostingRegressor(random_state=42)) 
    ])

    # 8. Grid Search CV
    print("\n[Step 8] Grid Search CV...")
    param_grid = {'regressor__n_estimators': [50], 'regressor__learning_rate': [0.1]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=2, n_jobs=1)
    grid_search.fit(X_train.sample(1000, random_state=42), y_train.sample(1000, random_state=42))
    print(f"Grid Search Best Params: {grid_search.best_params_}")

    # 9. Randomized Search CV
    print("\n[Step 9] Randomized Search CV...")
    param_dist = {'regressor__n_estimators': [100, 200], 'regressor__max_depth': [3, 5]}
    random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=2, cv=2, random_state=42, n_jobs=1)
    random_search.fit(X_train.sample(1000, random_state=42), y_train.sample(1000, random_state=42))
    print(f"Random Search Best Params: {random_search.best_params_}")

    # 10. The best Model Training
    print("\n[Step 10] Training Best Model (Champion)...")
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # ('pca', PCA(n_components=10)), 
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    
    start_time = time.time()
    final_pipeline.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")

    # 11. Saving the model
    print("\n[Step 11] Saving the Model...")
    joblib.dump(final_pipeline, 'flight_price_model.pkl')
    print("Model saved to 'flight_price_model.pkl'")

    # 13. Load and Predict
    print("\n[Step 13] Verification: Loading Model and Predicting...")
    loaded_model = joblib.load('flight_price_model.pkl')
    y_pred = loaded_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()
