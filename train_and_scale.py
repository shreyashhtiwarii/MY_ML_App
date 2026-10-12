import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

def run_pipeline(data_path):
    # 1. Load Data
    df = pd.read_csv(data_path)
    ml_data = df.dropna(subset=['year', 'figure']).copy()

    # 2. Outlier Capping (99th Percentile)
    cap_value = ml_data['figure'].quantile(0.99)
    ml_data['figure_capped'] = ml_data['figure'].clip(upper=cap_value)

    # 3. Features & Target
    features = ['year', 'university_id', 'category', 'seat_type']
    X = ml_data[features]
    y = ml_data['figure_capped']

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Target Encoding
    encoding_maps = {}
    X_train_encoded = X_train[['year']].copy()
    X_test_encoded = X_test[['year']].copy()

    for col in ['university_id', 'category', 'seat_type']:
        means = y_train.groupby(X_train[col]).mean()
        global_mean = y_train.mean()

        # Save mappings
        encoding_maps[col] = {
            'mappings': means.to_dict(),
            'global_mean': float(global_mean)
        }

        # Transform
        X_train_encoded[col + '_encoded'] = X_train[col].map(means).fillna(global_mean)
        X_test_encoded[col + '_encoded'] = X_test[col].map(means).fillna(global_mean)

    encoding_maps['unique_values'] = {
        'university_id': sorted(X_train['university_id'].unique().tolist()),
        'category': sorted(X_train['category'].unique().tolist()),
        'seat_type': sorted(X_train['seat_type'].unique().tolist())
    }

    # 6. Apply Feature Scaling (Crucial for balancing 'year' vs. encoded categories)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # 7. Model Training (Adjusted hyperparams to capture Year sensitivity)
    model = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 8. Save Artifacts
    joblib.dump(model, 'improved_model.pkl')
    joblib.dump(encoding_maps, 'encoding_maps.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model pipeline completed! Saved model, encoding maps, and scaler.")

if __name__ == '__main__':
    run_pipeline('/content/cutoffs.csv')