import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def build_and_evaluate(data_path):
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    ml_data = df.dropna(subset=['year', 'figure']).copy()
    
    # Cap extreme outliers at the 99th percentile
    cap_value = ml_data['figure'].quantile(0.99)
    ml_data['figure_capped'] = ml_data['figure'].clip(upper=cap_value)
    
    # Feature columns and target vector
    features = ['year', 'university_id', 'category', 'seat_type']
    X = ml_data[features]
    y = ml_data['figure_capped']
    
    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Implement target encoding to represent categoricals numerically
    X_train_encoded = X_train[['year']].copy()
    X_test_encoded = X_test[['year']].copy()
    
    for col in ['university_id', 'category', 'seat_type']:
        means = y_train.groupby(X_train[col]).mean()
        global_mean = y_train.mean()
        
        X_train_encoded[col + '_encoded'] = X_train[col].map(means).fillna(global_mean)
        X_test_encoded[col + '_encoded'] = X_test[col].map(means).fillna(global_mean)
        
    # Build and fit the optimized Random Forest Regressor
    print("Training the optimized Random Forest Regressor model...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_encoded, y_train)
    
    # Evaluate performance
    preds = rf_model.predict(X_test_encoded)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"
--- Model Training Completed ---")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")
    
    # Save final model state
    joblib.dump(rf_model, 'improved_model.pkl')
    print("
Model successfully saved to 'improved_model.pkl'")

if __name__ == '__main__':
    build_and_evaluate('/content/cutoffs.csv')