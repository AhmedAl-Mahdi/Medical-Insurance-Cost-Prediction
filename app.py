"""
Medical Insurance Cost Prediction Dashboard

A Streamlit application for exploring medical insurance data, training predictive models,
and making predictions on insurance costs and charge categories.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'regression_model' not in st.session_state:
    st.session_state.regression_model = None
if 'classification_model' not in st.session_state:
    st.session_state.classification_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'charge_category_encoder' not in st.session_state:
    st.session_state.charge_category_encoder = None

def load_data():
    """Load and preprocess the insurance dataset"""
    try:
        data = pd.read_csv('insurance.csv')
        st.session_state.data = data
        st.session_state.data_loaded = True
        return data
    except FileNotFoundError:
        st.error("Insurance dataset not found. Please ensure 'insurance.csv' is in the application directory.")
        return None

def categorize_charge(cost):
    """Categorize insurance charge based on cost thresholds"""
    # These thresholds are approximate quartiles based on typical insurance data
    if cost < 4000:
        return "Low"
    elif cost < 9000:
        return "Medium" 
    elif cost < 16000:
        return "High"
    else:
        return "Very High"
    """Preprocess the data for model training"""
    processed_data = data.copy()
    
    # Initialize label encoders
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    # Encode categorical variables
    processed_data['sex'] = le_sex.fit_transform(processed_data['sex'])
    processed_data['smoker'] = le_smoker.fit_transform(processed_data['smoker'])
    processed_data['region'] = le_region.fit_transform(processed_data['region'])
    
    # Store encoders in session state
    st.session_state.label_encoders = {
        'sex': le_sex,
        'smoker': le_smoker,
        'region': le_region
    }
    
    # Create charge categories based on quartiles
    processed_data['charge_category'] = pd.qcut(processed_data['charges'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Encode charge categories
    le_charge_category = LabelEncoder()
    processed_data['charge_category'] = le_charge_category.fit_transform(processed_data['charge_category'])
    st.session_state.charge_category_encoder = le_charge_category
    
    return processed_data

def build_regression_model(input_shape, optimizer_type, learning_rate, momentum=None):
    """Build the regression model for cost prediction"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1)  # Linear output for regression
    ])
    
    # Create new optimizer instance
    if optimizer_type == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_classification_model(input_shape, num_classes, optimizer_type, learning_rate, momentum=None):
    """Build the classification model for charge category prediction"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
    ])
    
    # Create new optimizer instance
    if optimizer_type == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main application function"""
    st.title("🏥 Medical Insurance Cost Prediction Dashboard")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📊 Data Overview & EDA", "🤖 Model Training", "🔮 Predictions"]
    )
    
    # Load data
    if not st.session_state.data_loaded:
        data = load_data()
        if data is None:
            return
    else:
        data = st.session_state.data
    
    # Page routing
    if page == "📊 Data Overview & EDA":
        show_eda_page(data)
    elif page == "🤖 Model Training":
        show_training_page(data)
    elif page == "🔮 Predictions":
        show_prediction_page(data)

def show_eda_page(data):
    """Display exploratory data analysis page"""
    st.header("📊 Data Overview & Exploratory Data Analysis")
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Features", len(data.columns) - 1)
    with col3:
        st.metric("Average Cost", f"${data['charges'].mean():,.2f}")
    with col4:
        st.metric("Max Cost", f"${data['charges'].max():,.2f}")
    
    # Display dataset
    with st.expander("📋 Dataset Preview", expanded=False):
        st.dataframe(data)
    
    # Basic statistics
    with st.expander("📈 Statistical Summary", expanded=False):
        st.write(data.describe())
    
    # Visualizations
    st.subheader("📊 Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of charges
        fig = px.histogram(data, x='charges', nbins=30, title='Distribution of Insurance Charges')
        st.plotly_chart(fig, use_container_width=True)
        
        # Age vs Charges
        fig = px.scatter(data, x='age', y='charges', title='Age vs Insurance Charges')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Charges by smoker status
        fig = px.box(data, x='smoker', y='charges', title='Insurance Charges by Smoker Status')
        st.plotly_chart(fig, use_container_width=True)
        
        # BMI vs Charges colored by smoker
        fig = px.scatter(data, x='bmi', y='charges', color='smoker', title='BMI vs Insurance Charges')
        st.plotly_chart(fig, use_container_width=True)

def show_training_page(data):
    """Display model training page"""
    st.header("🤖 Model Training Configuration")
    
    # Training configuration
    st.subheader("🔧 Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Optimizer selection
        optimizer_choice = st.selectbox(
            "Select Optimizer",
            ["Adam", "SGD"],
            index=0,
            help="Choose the optimizer for model training"
        )
    
    with col2:
        # Learning rate
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            step=0.0001,
            format="%.4f",
            help="Learning rate for the optimizer"
        )
    
    with col3:
        # Momentum (only for SGD)
        if optimizer_choice == "SGD":
            momentum = st.slider(
                "Momentum",
                min_value=0.0,
                max_value=0.95,
                value=0.9,
                step=0.05,
                help="Momentum for SGD optimizer"
            )
        else:
            momentum = None
    
    # Additional training parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input("Epochs", min_value=10, max_value=200, value=100, step=10)
    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    with col3:
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    
    # Create optimizer instance
    optimizer_display = ""
    if optimizer_choice == "Adam":
        optimizer_display = f"Adam (lr={learning_rate})"
    else:
        optimizer_display = f"SGD (lr={learning_rate}, momentum={momentum})"
    
    # Model Configuration Summary
    with st.expander("📋 Model Configuration Summary", expanded=False):
        st.write("**Optimizer Configuration:**")
        st.write(f"- {optimizer_display}")
        
        st.write("**Regression Model Configuration:**")
        st.write("- Hidden Activations: ReLU")
        st.write("- Output Activation: Linear")
        st.write("- Loss Function: MSE")
        st.write("- Metrics: MAE")
        
        st.write("**Classification Model Configuration:**")
        st.write("- Hidden Activations: ReLU")
        st.write("- Output Activation: Softmax")
        st.write("- Loss Function: Sparse Categorical Crossentropy")
        st.write("- Metrics: Accuracy")
    
    # Train models button
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            try:
                # Clear any existing TensorFlow sessions
                tf.keras.backend.clear_session()
                
                # Preprocess data
                processed_data = preprocess_data(data)
                
                # Prepare data for regression
                X = processed_data.drop(['charges', 'charge_category'], axis=1)
                y_reg = processed_data['charges']
                y_class = processed_data['charge_category']
                
                # Split data
                X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
                    X, y_reg, y_class, test_size=test_size, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Store scaler
                st.session_state.scaler = scaler
                
                # Build and train regression model
                reg_model = build_regression_model(
                    X_train_scaled.shape[1], 
                    optimizer_choice, 
                    learning_rate, 
                    momentum
                )
                reg_history = reg_model.fit(
                    X_train_scaled, y_reg_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=0
                )
                
                # Build and train classification model
                class_model = build_classification_model(
                    X_train_scaled.shape[1], 
                    len(st.session_state.charge_category_encoder.classes_),
                    optimizer_choice,
                    learning_rate,
                    momentum
                )
                class_history = class_model.fit(
                    X_train_scaled, y_class_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=0
                )
                
                # Store models
                st.session_state.regression_model = reg_model
                st.session_state.classification_model = class_model
                st.session_state.models_trained = True
                
                # Evaluate models
                reg_pred = reg_model.predict(X_test_scaled, verbose=0)
                reg_mse = mean_squared_error(y_reg_test, reg_pred.flatten())
                reg_mae = mean_absolute_error(y_reg_test, reg_pred.flatten())
                reg_r2 = r2_score(y_reg_test, reg_pred.flatten())
                
                class_pred_prob = class_model.predict(X_test_scaled, verbose=0)
                class_pred = np.argmax(class_pred_prob, axis=1)
                class_acc = accuracy_score(y_class_test, class_pred)
                
                st.success("Models trained successfully!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Regression Model Results")
                    st.metric("MSE", f"{reg_mse:,.2f}")
                    st.metric("MAE", f"{reg_mae:,.2f}")
                    st.metric("R² Score", f"{reg_r2:.3f}")
                
                with col2:
                    st.subheader("🎯 Classification Model Results")
                    st.metric("Accuracy", f"{class_acc:.3f}")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

def show_prediction_page(data):
    """Display prediction page"""
    st.header("🔮 Make Predictions")
    
    # Check if models are trained
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first in the Model Training page.")
        
        # Add a demo button to show the interface
        if st.button("🎯 Show Demo Prediction Interface", help="Click to see how the prediction interface would look"):
            st.session_state.demo_mode = True
    
    # Show prediction interface if models are trained OR in demo mode
    if st.session_state.models_trained or st.session_state.get('demo_mode', False):
        if st.session_state.get('demo_mode', False):
            st.info("🎭 **Demo Mode**: This shows how the prediction interface works. Train models to make real predictions.")
        
        st.subheader("📝 Enter Patient Information")
        
        # Input fields
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            sex = st.selectbox("Sex", ["female", "male"])
        
        with col2:
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
        
        with col3:
            smoker = st.selectbox("Smoker", ["no", "yes"])
            region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        
        # Predict button
        if st.button("🔮 Predict", type="primary"):
            if st.session_state.get('demo_mode', False):
                # Demo mode - show mock predictions
                st.subheader("🎯 Prediction Results (Demo)")
                
                # Mock predictions for demo
                import random
                random.seed(42)
                mock_cost = 8000 + (age * 100) + (1000 if smoker == "yes" else 0) + (bmi * 50)
                mock_categories = ["Low", "Medium", "High", "Very High"]
                mock_category = mock_categories[2 if smoker == "yes" else 1]  # Higher if smoker
                mock_probabilities = [0.1, 0.3, 0.5, 0.1] if smoker == "no" else [0.05, 0.15, 0.6, 0.2]
                
                # Get regression-derived category
                regression_category = categorize_charge(mock_cost)
                
                # Display predictions as metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Predicted Insurance Cost",
                        f"${mock_cost:,.2f}",
                        help="Predicted insurance charge amount"
                    )
                    st.caption(f"*Regression-derived category: {regression_category}*")
                
                with col2:
                    st.metric(
                        "Predicted Charge Category",
                        mock_category,
                        help="Category based on charge quartiles"
                    )
                
                # Classification probabilities bar chart
                st.subheader("📊 Classification Probabilities")
                prob_df = pd.DataFrame({
                    'Category': mock_categories,
                    'Probability': mock_probabilities
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Category', 
                    y='Probability',
                    title='Probability Distribution for Charge Categories',
                    color='Probability',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Real prediction mode
                try:
                    # Prepare input data
                    input_data = pd.DataFrame({
                        'age': [age],
                        'sex': [sex],
                        'bmi': [bmi],
                        'children': [children],
                        'smoker': [smoker],
                        'region': [region]
                    })
                    
                    # Encode categorical variables
                    input_data['sex'] = st.session_state.label_encoders['sex'].transform(input_data['sex'])
                    input_data['smoker'] = st.session_state.label_encoders['smoker'].transform(input_data['smoker'])
                    input_data['region'] = st.session_state.label_encoders['region'].transform(input_data['region'])
                    
                    # Scale input data
                    input_scaled = st.session_state.scaler.transform(input_data)
                    
                    # Make predictions
                    cost_pred = st.session_state.regression_model.predict(input_scaled, verbose=0)[0][0]
                    class_pred_prob = st.session_state.classification_model.predict(input_scaled, verbose=0)[0]
                    class_pred = np.argmax(class_pred_prob)
                    
                    # Get category name
                    category_name = st.session_state.charge_category_encoder.inverse_transform([class_pred])[0]
                    
                    # Get regression-derived category
                    regression_category = categorize_charge(cost_pred)
                    
                    st.subheader("🎯 Prediction Results")
                    
                    # Display predictions as metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Predicted Insurance Cost",
                            f"${cost_pred:,.2f}",
                            help="Predicted insurance charge amount"
                        )
                        st.caption(f"*Regression-derived category: {regression_category}*")
                    
                    with col2:
                        st.metric(
                            "Predicted Charge Category",
                            category_name,
                            help="Category based on charge quartiles"
                        )
                    
                    # Classification probabilities bar chart
                    st.subheader("📊 Classification Probabilities")
                    prob_df = pd.DataFrame({
                        'Category': st.session_state.charge_category_encoder.classes_,
                        'Probability': class_pred_prob
                    })
                    
                    fig = px.bar(
                        prob_df, 
                        x='Category', 
                        y='Probability',
                        title='Probability Distribution for Charge Categories',
                        color='Probability',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()