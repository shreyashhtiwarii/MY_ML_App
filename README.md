# University Cutoff Prediction & Streamlit Dashboard

This project processes university admission cutoff data, applies robust data preprocessing, performs target encoding to transform critical categorical attributes, and trains a Random Forest regression model to predict cutoff scores. An interactive Streamlit dashboard is included to make on-demand predictions.

## Project Components

- `app.py`: The frontend application code powered by Streamlit.
- `improved_model.pkl`: The trained Random Forest model (R² score ~0.63).
- `encoding_maps.pkl`: Exported target-encoding mappings for categorical input transformations.
- `requirements.txt`: Python dependencies required to run or deploy the project.

## Machine Learning Pipeline

1. **Outlier Mitigation**: Cutoff figures were cleaned and capped at the 99th percentile (`figure_capped`) to minimize regression skew.
2. **Feature Engineering**: Encoded high-cardinality predictors (`university_id`, `category`, `seat_type`) with target encoding using cross-validation guards to prevent data leakage.
3. **Evaluation**: Transitioning from a single predictor (`year`) to target-encoded features dramatically increased model accuracy ($R^2$ improved from `0.03` to `0.63`).

## Local Installation & Setup

1. Clone or download this project workspace into a local directory.
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit server locally:
   ```bash
   streamlit run app.py
   ```

## Deployment (Streamlit Cloud)

To host this interactive predictor on the web:
1. Upload `app.py`, `improved_model.pkl`, `encoding_maps.pkl`, and `requirements.txt` to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New App**, point to your repository and branch, and deploy!