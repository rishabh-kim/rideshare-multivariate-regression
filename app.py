import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🚗 Rideshare Price Predictor", layout="wide")

st.title("🚗 Rideshare Price Predictor")
st.markdown("Predict rideshare prices using machine learning powered by multivariate regression")

# Load data with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('rideshare_kaggle.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please ensure 'rideshare_kaggle.csv' exists.")
        st.stop()

df = load_data()

# Display what columns we have
st.sidebar.info(f"✅ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Select numeric columns only (ignore text/datetime)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# If not enough numeric columns, try to identify them by name patterns
if len(numeric_cols) < 2:
    # Common numeric column patterns to look for
    numeric_patterns = ['price', 'distance', 'surge', 'temperature', 'latitude', 'longitude',
                       'hour', 'day', 'month', 'timestamp', 'humidity', 'wind', 'uv']
    
    for col in df.columns:
        if col.lower() in numeric_patterns and df[col].dtype == 'object':
            # Try to convert
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].notna().sum() > len(df) * 0.5:  # If >50% values can be numeric
                numeric_cols.append(col)
    
    numeric_cols = list(set(numeric_cols))

st.sidebar.write(f"**Numeric Features Found**: {len(numeric_cols)}")
if len(numeric_cols) < 2:
    st.error(f"❌ Not enough numeric features! Found only {len(numeric_cols)} numeric columns.")
    st.write(f"Available columns: {df.columns.tolist()}")
    st.write(f"Numeric columns detected: {numeric_cols}")
    st.stop()

# Identify potential target and features
target_col = 'price' if 'price' in numeric_cols else numeric_cols[-1] if numeric_cols else None

if target_col is None:
    st.error("❌ Could not identify a target column. Need 'price' or numeric columns.")
    st.stop()

# Show data overview
with st.expander("📊 Dataset Overview"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Numeric Columns", len(numeric_cols))
    col3.metric("Target Column", target_col)
    
    st.write("**Column Names:**")
    st.write(numeric_cols)
    st.write("**First 5 rows:**")
    st.dataframe(df.head())

# Prepare data for modeling
feature_cols = [col for col in numeric_cols if col != target_col]

if len(feature_cols) < 1:
    st.error("Need at least 1 feature column besides the target")
    st.stop()

# Clean data
df_clean = df[feature_cols + [target_col]].dropna()
st.sidebar.write(f"**Clean records**: {len(df_clean):,} (after removing NaN)")

X = df_clean[feature_cols].values
y = df_clean[target_col].values

# Split data
split_idx = int(len(df_clean) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
@st.cache_resource
def train_model():
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model

model = train_model()

# Get scores
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Train R² Score", f"{train_score:.4f}")
col2.metric("Test R² Score", f"{test_score:.4f}")
col3.metric("Train/Test Split", f"{split_idx}/{len(df_clean) - split_idx}")

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Make Prediction", "📈 Data Explorer", "🤖 Model Performance", "📊 Feature Analysis"])

# TAB 1: PREDICTIONS
with tab1:
    st.subheader("Make a Price Prediction")
    
    prediction_input = {}
    cols = st.columns(len(feature_cols))
    
    for idx, col_name in enumerate(feature_cols):
        with cols[idx % len(cols)]:
            min_val = X[:, idx].min()
            max_val = X[:, idx].max()
            mean_val = X[:, idx].mean()
            
            val = st.slider(
                col_name.replace('_', ' ').title(),
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=float((max_val - min_val) / 100)
            )
            prediction_input[col_name] = val
    
    # Make prediction
    input_array = np.array([prediction_input[col] for col in feature_cols]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    predicted_value = model.predict(input_scaled)[0]
    
    st.markdown("---")
    st.markdown("### 🎯 Predicted Price")
    st.markdown(f"## **${predicted_value:.2f}**" if target_col == 'price' else f"## **{predicted_value:.2f}**")

# TAB 2: DATA EXPLORER
with tab2:
    st.subheader("Dataset Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(y, bins=50, color='#3498DB', edgecolor='black', alpha=0.7)
        ax.set_xlabel(target_col.title(), fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{target_col.title()} Distribution', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        if len(feature_cols) >= 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(X[:, 0], y, alpha=0.5, s=20, color='#E74C3C')
            ax.set_xlabel(feature_cols[0].title(), fontweight='bold')
            ax.set_ylabel(target_col.title(), fontweight='bold')
            ax.set_title(f'{feature_cols[0].title()} vs {target_col.title()}', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(df_clean.head(20), use_container_width=True)

# TAB 3: MODEL PERFORMANCE
with tab3:
    st.subheader("Model Performance Analysis")
    
    y_pred_test = model.predict(X_test_scaled)
    residuals = y_test - y_pred_test
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("Train R²", f"{train_score:.4f}")
    col4.metric("Test R²", f"{test_score:.4f}")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y_test, y_pred_test, alpha=0.6, s=20, color='#3498DB')
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('Actual', fontweight='bold')
        ax.set_ylabel('Predicted', fontweight='bold')
        ax.set_title('Predictions vs Actual', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y_pred_test, residuals, alpha=0.6, s=20, color='#E74C3C')
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Residuals', fontweight='bold')
        ax.set_title('Residual Plot', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    with col3:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(residuals, bins=50, color='#9B59B6', edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Residual Distribution', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

# TAB 4: FEATURE ANALYSIS
with tab4:
    st.subheader("Feature Importance & Coefficients")
    
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_,
        'Abs Coefficient': np.abs(model.coef_)
    }).sort_values('Abs Coefficient', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#27AE60' if x > 0 else '#E74C3C' for x in coef_df['Coefficient']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black')
    ax.set_xlabel('Coefficient Value', fontweight='bold')
    ax.set_title('Feature Coefficients (Importance)', fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Coefficient Table")
    st.dataframe(coef_df[['Feature', 'Coefficient']], use_container_width=True)
    
    st.markdown("---")
    st.subheader("Correlation with Target")
    
    corr_data = []
    for col in feature_cols:
        col_idx = feature_cols.index(col)
        corr = np.corrcoef(X[:, col_idx], y)[0, 1]
        corr_data.append({'Feature': col, 'Correlation': corr})
    
    corr_df = pd.DataFrame(corr_data).sort_values('Correlation', key=abs, ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_corr = ['#27AE60' if x > 0 else '#E74C3C' for x in corr_df['Correlation']]
    ax.barh(corr_df['Feature'], corr_df['Correlation'], color=colors_corr, edgecolor='black')
    ax.set_xlabel('Correlation with Target', fontweight='bold')
    ax.set_title('Feature Correlations', fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    st.pyplot(fig)

st.markdown("---")
st.markdown("🚗 **Rideshare Price Predictor** | Built with Streamlit & Scikit-learn | ML Regression")
