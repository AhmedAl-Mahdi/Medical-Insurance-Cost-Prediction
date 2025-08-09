import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def inject_global_styles():
    """Inject medical-themed CSS styles"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for theming */
    :root {
        --primary-blue: #2E86AB;
        --primary-teal: #A23B72;
        --accent-red: #F18F01;
        --light-blue: #F0F8FF;
        --soft-gray: #F8F9FA;
        --dark-gray: #2C3E50;
        --success-green: #28A745;
        --gradient-bg: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Dark mode variables */
    [data-theme="dark"] {
        --primary-blue: #4A9ECC;
        --primary-teal: #C855A0;
        --light-blue: #1E3A5F;
        --soft-gray: #2D3748;
        --dark-gray: #F7FAFC;
        --gradient-bg: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
    }
    
    /* Dark mode body styling */
    [data-theme="dark"] .stApp {
        background: var(--gradient-bg);
        color: var(--dark-gray);
    }
    
    [data-theme="dark"] .card {
        background: var(--soft-gray);
        color: var(--dark-gray);
        border-left-color: var(--primary-blue);
    }
    
    [data-theme="dark"] .metric-card {
        background: var(--soft-gray);
        color: var(--dark-gray);
        border-top-color: var(--primary-blue);
    }
    
    [data-theme="dark"] .clinical-context {
        background: #2D3748;
        border-color: #4A5568;
        color: var(--dark-gray);
    }
    
    /* Global font */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--gradient-bg);
    }
    
    /* Header styling */
    .medical-header {
        background: linear-gradient(90deg, var(--primary-blue), var(--primary-teal));
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .medical-header h1 {
        margin: 0;
        font-weight: 600;
        font-size: 2.5rem;
    }
    
    .medical-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-blue);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    /* Section titles */
    .section-title {
        color: var(--primary-blue);
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-blue);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-top: 4px solid var(--primary-blue);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-blue);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: var(--dark-gray);
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    /* Category chips */
    .chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.85rem;
        margin: 0.25rem;
        color: white;
    }
    
    .chip-low { background-color: var(--success-green); }
    .chip-medium { background-color: #FFC107; }
    .chip-high { background-color: #FF6B35; }
    .chip-very-high { background-color: var(--accent-red); }
    
    /* Clinical context box */
    .clinical-context {
        background: #FFF9E6;
        border: 1px solid #FFE066;
        border-left: 4px solid #FFC107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .clinical-context .warning-icon {
        color: #FF8C00;
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, var(--primary-blue), var(--primary-teal));
        margin: 2rem 0;
        border-radius: 1px;
    }
    
    /* Button enhancements */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 2px solid var(--primary-blue);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(46, 134, 171, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background-color: var(--primary-blue);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--light-blue), white);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        border: 1px solid #E2E8F0;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-blue);
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--light-blue), white);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .medical-header h1 {
            font-size: 2rem;
        }
        
        .medical-header p {
            font-size: 1rem;
        }
        
        .metric-card {
            margin-bottom: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_medical_header():
    """Create a medical-themed header section"""
    st.markdown("""
    <div class="medical-header">
        <h1>🩺 Medical Insurance Cost Predictor</h1>
        <p>Advanced Analytics for Healthcare Cost Estimation and Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

def create_clinical_context():
    """Create clinical context disclaimer"""
    st.markdown("""
    <div class="clinical-context">
        <span class="warning-icon">⚠️</span>
        <strong>Clinical Context:</strong> This tool provides educational cost predictions based on statistical models. 
        Predictions should not be used for actual medical decision-making or insurance purposes. 
        Always consult with qualified healthcare and insurance professionals for real-world applications.
    </div>
    """, unsafe_allow_html=True)

def create_category_chip(category):
    """Create a styled category chip"""
    chip_class = f"chip-{category.lower().replace(' ', '-')}"
    return f'<span class="chip {chip_class}">{category}</span>'

@st.cache_data
def load_data():
    """Load and cache the insurance dataset"""
    data = pd.read_csv('insurance.csv')
    return data

def preprocess_data(data):
    """Preprocess the data for modeling"""
    processed_data = data.copy()
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    processed_data['sex'] = le_sex.fit_transform(processed_data['sex'])
    processed_data['smoker'] = le_smoker.fit_transform(processed_data['smoker'])
    processed_data['region'] = le_region.fit_transform(processed_data['region'])
    
    # Create charge categories for classification
    processed_data['charge_category'] = pd.cut(
        processed_data['charges'],
        bins=[0, 5000, 15000, 30000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    return processed_data, le_sex, le_smoker, le_region

def eda_section(data):
    """Exploratory Data Analysis section"""
    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(data)}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data['age'].mean():.1f}</div>
            <div class="metric-label">Avg Age</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${data['charges'].mean():.0f}</div>
            <div class="metric-label">Avg Cost</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        smoker_pct = (data['smoker'] == 'yes').mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{smoker_pct:.1f}%</div>
            <div class="metric-label">Smokers</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Correlation Heatmap", "Feature Relationships"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data['charges'], kde=True, ax=ax, color='#2E86AB')
            ax.set_title('Distribution of Medical Charges')
            ax.set_xlabel('Charges ($)')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=data, x='smoker', y='charges', ax=ax, palette=['#2E86AB', '#F18F01'])
            ax.set_title('Charges by Smoking Status')
            st.pyplot(fig)
    
    with tab2:
        numeric_data = data.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='Blues', ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=data, x='age', y='charges', hue='smoker', ax=ax, palette=['#2E86AB', '#F18F01'])
            ax.set_title('Age vs Charges by Smoking Status')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=data, x='bmi', y='charges', hue='smoker', ax=ax, palette=['#2E86AB', '#F18F01'])
            ax.set_title('BMI vs Charges by Smoking Status')
            st.pyplot(fig)

def model_training_section(processed_data):
    """Model training section"""
    st.markdown('<div class="section-title">🧬 Model Training & Evaluation</div>', unsafe_allow_html=True)
    
    # Training controls in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>🏗️ Model Architecture</h4>
        </div>
        """, unsafe_allow_html=True)
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Linear Regression", "Random Forest", "Neural Network"]
        )
        
        if model_type == "Neural Network":
            hidden_layers = st.slider("Hidden Layers", 1, 3, 2)
            neurons = st.slider("Neurons per Layer", 16, 128, 64)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4>⚙️ Training Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        random_state = st.number_input("Random State", value=42)
        
        if model_type == "Neural Network":
            epochs = st.slider("Training Epochs", 50, 300, 100)
            learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)
    
    if st.button("🚀 Train Models", type="primary"):
        train_models(processed_data, model_type, test_size, random_state)

def train_models(processed_data, model_type, test_size, random_state):
    """Train and evaluate models"""
    # Prepare data
    features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    X = processed_data[features]
    y_reg = processed_data['charges']
    y_class = processed_data['charge_category']
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store scalers and feature names in session state
    st.session_state.scaler = scaler
    st.session_state.feature_names = features
    
    # Train regression model
    with st.spinner("Training regression model..."):
        if model_type == "Linear Regression":
            reg_model = LinearRegression()
            reg_model.fit(X_train_scaled, y_reg_train)
        elif model_type == "Random Forest":
            reg_model = RandomForestRegressor(random_state=random_state)
            reg_model.fit(X_train_scaled, y_reg_train)
        else:  # Neural Network
            reg_model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            reg_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            reg_model.fit(X_train_scaled, y_reg_train, epochs=100, verbose=0, validation_split=0.2)
    
    # Store model in session state
    st.session_state.regression_model = reg_model
    st.session_state.model_type = model_type
    
    # Make predictions
    if model_type == "Neural Network":
        y_pred = reg_model.predict(X_test_scaled).flatten()
    else:
        y_pred = reg_model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_reg_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_reg_test, y_pred)
    mae = mean_absolute_error(y_reg_test, y_pred)
    
    # Display results
    st.success("✅ Model training completed!")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${rmse:.0f}</div>
            <div class="metric-label">RMSE</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{r2:.3f}</div>
            <div class="metric-label">R² Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${mae:.0f}</div>
            <div class="metric-label">MAE</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${mse:.0f}</div>
            <div class="metric-label">MSE</div>
        </div>
        """, unsafe_allow_html=True)

def prediction_section():
    """Prediction section with enhanced UI"""
    st.markdown('<div class="section-title">💊 Insurance Cost Prediction</div>', unsafe_allow_html=True)
    
    if 'regression_model' not in st.session_state:
        st.warning("⚠️ Please train a model first in the Model Training section.")
        return
    
    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 80, 30)
            sex = st.selectbox("Sex", ["female", "male"])
            
        with col2:
            bmi = st.slider("BMI", 15.0, 50.0, 25.0)
            children = st.slider("Number of Children", 0, 5, 0)
            
        with col3:
            smoker = st.selectbox("Smoker", ["no", "yes"])
            region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        
        submitted = st.form_submit_button("🔮 Predict Cost", type="primary")
    
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex == 'male' else 0],
            'bmi': [bmi],
            'children': [children],
            'smoker': [1 if smoker == 'yes' else 0],
            'region': [{'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}[region]]
        })
        
        # Scale input
        input_scaled = st.session_state.scaler.transform(input_data)
        
        # Make prediction
        if st.session_state.model_type == "Neural Network":
            prediction = st.session_state.regression_model.predict(input_scaled)[0][0]
        else:
            prediction = st.session_state.regression_model.predict(input_scaled)[0]
        
        # Determine category
        if prediction < 5000:
            category = "Low"
        elif prediction < 15000:
            category = "Medium"
        elif prediction < 30000:
            category = "High"
        else:
            category = "Very High"
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <h3>💰 Predicted Cost</h3>
                <div style="text-align: center;">
                    <div style="font-size: 3rem; font-weight: bold; color: #2E86AB; margin: 1rem 0;">
                        ${prediction:,.0f}
                    </div>
                    <p style="color: #666; font-style: italic;">
                        Annual insurance premium estimate
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            chip_html = create_category_chip(category)
            st.markdown(f"""
            <div class="card">
                <h3>📊 Cost Category</h3>
                <div style="text-align: center; margin: 2rem 0;">
                    {chip_html}
                </div>
                <p style="color: #666; font-style: italic; text-align: center;">
                    Risk classification based on predicted cost
                </p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application"""
    # Page configuration
    st.set_page_config(
        page_title="Medical Insurance Cost Predictor",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject styles
    inject_global_styles()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎨 Display Options")
        
        # Dark mode toggle
        dark_mode = st.checkbox("🌙 Dark Mode", key="dark_mode")
        if dark_mode:
            st.markdown("""
            <script>
            document.documentElement.setAttribute('data-theme', 'dark');
            </script>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <script>
            document.documentElement.removeAttribute('data-theme');
            </script>
            """, unsafe_allow_html=True)
        
        # Show clinical context
        show_context = st.checkbox("⚠️ Show Clinical Context", value=True)
    
    # Header
    create_medical_header()
    
    # Clinical context
    if show_context:
        create_clinical_context()
    
    # Load data
    data = load_data()
    processed_data, le_sex, le_smoker, le_region = preprocess_data(data)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Exploratory Analysis", "🧬 Model Training", "💊 Predictions"])
    
    with tab1:
        eda_section(data)
    
    with tab2:
        model_training_section(processed_data)
    
    with tab3:
        prediction_section()

if __name__ == "__main__":
    main()