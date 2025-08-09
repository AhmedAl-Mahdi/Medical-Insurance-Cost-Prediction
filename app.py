import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD

# Set page config
st.set_page_config(
    page_title="Medical Insurance Cost Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}

def load_default_data():
    """Load the default insurance.csv dataset"""
    try:
        data = pd.read_csv('insurance.csv')
        return data
    except FileNotFoundError:
        st.error("Default insurance.csv file not found. Please upload a dataset.")
        return None

def get_charge_category_from_cost(cost):
    """Convert insurance cost to charge category"""
    if cost < 5000:
        return "Low", "💚"
    elif cost < 15000:
        return "Medium", "💛"
    elif cost < 30000:
        return "High", "🧡"
    else:
        return "Very High", "❤️"

def get_cost_range_from_category(category):
    """Get cost range for a given category"""
    cost_ranges = {
        "Low": (0, 5000),
        "Medium": (5000, 15000), 
        "High": (15000, 30000),
        "Very High": (30000, 100000)
    }
    return cost_ranges.get(category, (0, 100000))

def preprocess_data(data, target_col='charges'):
    """Preprocess the data with encoding and scaling"""
    processed_data = data.copy()
    
    # Store original categorical values for later reference
    categorical_cols = processed_data.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # Initialize encoders
    encoders = {}
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        processed_data[col] = le.fit_transform(processed_data[col])
        encoders[col] = le
    
    # Create charge categories for classification
    if target_col == 'charges':
        processed_data['charge_category'] = pd.qcut(processed_data['charges'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        le_charge = LabelEncoder()
        processed_data['charge_category_encoded'] = le_charge.fit_transform(processed_data['charge_category'])
        encoders['charge_category'] = le_charge
    
    return processed_data, encoders

def create_eda_section(data):
    """Create the EDA section with interactive visualizations"""
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if data is None:
        st.warning("Please load a dataset first.")
        return
    
    # Dataset Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Shape", f"{data.shape[0]} rows × {data.shape[1]} cols")
    with col2:
        st.metric("Missing Values", data.isnull().sum().sum())
    with col3:
        st.metric("Numerical Columns", len(data.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Categorical Columns", len(data.select_dtypes(include=['object']).columns))
    
    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    st.dataframe(data.describe(), use_container_width=True)
    
    # Missing Values Analysis
    if data.isnull().sum().sum() > 0:
        st.subheader("Missing Values")
        missing_data = data.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Count']
        missing_data = missing_data[missing_data['Missing Count'] > 0]
        st.dataframe(missing_data, use_container_width=True)
    
    # Interactive Visualizations
    st.subheader("Interactive Visualizations")
    
    # Column selection for visualizations
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    viz_tabs = st.tabs(["Distributions", "Relationships", "Categorical Analysis", "Correlation Matrix"])
    
    with viz_tabs[0]:
        st.write("**Distribution Analysis**")
        if numeric_cols:
            selected_col = st.selectbox("Select column for distribution", numeric_cols, key="dist_col")
            
            col1, col2 = st.columns(2)
            with col1:
                # Histogram
                fig_hist = px.histogram(data, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot
                fig_box = px.box(data, y=selected_col, title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig_box, use_container_width=True)
    
    with viz_tabs[1]:
        st.write("**Relationship Analysis**")
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X variable", numeric_cols, key="x_var")
            with col2:
                y_col = st.selectbox("Select Y variable", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="y_var")
            
            # Scatter plot
            fig_scatter = px.scatter(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with viz_tabs[2]:
        st.write("**Categorical Analysis**")
        if categorical_cols:
            selected_cat = st.selectbox("Select categorical column", categorical_cols, key="cat_col")
            
            col1, col2 = st.columns(2)
            with col1:
                # Count plot
                value_counts = data[selected_cat].value_counts()
                fig_count = px.bar(x=value_counts.index, y=value_counts.values, 
                                 title=f"Count of {selected_cat}",
                                 labels={'x': selected_cat, 'y': 'Count'})
                st.plotly_chart(fig_count, use_container_width=True)
            
            with col2:
                # Pie chart
                fig_pie = px.pie(values=value_counts.values, names=value_counts.index, 
                               title=f"Distribution of {selected_cat}")
                st.plotly_chart(fig_pie, use_container_width=True)
    
    with viz_tabs[3]:
        st.write("**Correlation Analysis**")
        if len(numeric_cols) > 1:
            # Correlation matrix
            correlation_matrix = data[numeric_cols].corr()
            
            fig_corr = px.imshow(correlation_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               title="Correlation Matrix",
                               color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_corr, use_container_width=True)

def create_preprocessing_section(data):
    """Create the preprocessing section"""
    st.markdown('<div class="section-header">🔧 Data Preprocessing</div>', unsafe_allow_html=True)
    
    if data is None:
        st.warning("Please load a dataset first.")
        return None, None
    
    # Feature selection
    st.subheader("Feature Selection")
    all_columns = data.columns.tolist()
    
    # Target column selection
    target_col = st.selectbox("Select target column", all_columns, 
                             index=all_columns.index('charges') if 'charges' in all_columns else 0)
    
    # Feature columns selection
    available_features = [col for col in all_columns if col != target_col]
    selected_features = st.multiselect("Select feature columns", available_features, 
                                     default=available_features)
    
    if not selected_features:
        st.error("Please select at least one feature column.")
        return None, None
    
    # Preprocessing options
    st.subheader("Preprocessing Options")
    
    col1, col2 = st.columns(2)
    with col1:
        encode_categorical = st.checkbox("Encode categorical variables", value=True)
        scale_features = st.checkbox("Scale numerical features", value=True)
    
    with col2:
        create_categories = st.checkbox("Create charge categories (for classification)", 
                                      value=True if target_col == 'charges' else False)
    
    if st.button("Apply Preprocessing"):
        try:
            # Preprocess data
            processed_data, encoders = preprocess_data(data, target_col)
            
            # Store in session state
            st.session_state.processed_data = processed_data
            st.session_state.encoders = encoders
            st.session_state.selected_features = selected_features
            st.session_state.target_col = target_col
            
            st.success("Data preprocessing completed!")
            
            # Show encoding information
            if encode_categorical:
                st.subheader("Encoding Information")
                for col, encoder in encoders.items():
                    if hasattr(encoder, 'classes_'):
                        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                        st.write(f"**{col}**: {mapping}")
            
            # Show processed data preview
            st.subheader("Processed Data Preview")
            st.dataframe(processed_data.head(), use_container_width=True)
            
            return processed_data, selected_features
            
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            return None, None
    
    return st.session_state.get('processed_data'), st.session_state.get('selected_features')

def create_model_training_section(processed_data, selected_features):
    """Create the model training section"""
    st.markdown('<div class="section-header">🤖 Model Training</div>', unsafe_allow_html=True)
    
    if processed_data is None or selected_features is None:
        st.warning("Please complete data preprocessing first.")
        return
    
    # Model type selection
    model_type = st.radio("Select model type", ["Regression", "Classification"])
    
    # Model parameters
    st.subheader("Model Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        epochs = st.slider("Epochs", 10, 500, 100)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    with col2:
        layer1_units = st.slider("First Layer Units", 16, 256, 64)
        layer2_units = st.slider("Second Layer Units", 8, 128, 32)
    
    with col3:
        learning_rate = st.selectbox("Learning Rate", [0.0001, 0.001, 0.01, 0.1], index=1)
        validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2)
    
    with col4:
        optimizer_choice = st.selectbox("Optimizer", ["Adam", "SGD"], index=0)
    
    # Display model architecture information
    st.subheader("Model Architecture Information")
    
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.info(f"""
        **Neural Network Architecture:**
        - Input Layer: {len(selected_features) if selected_features else 'N/A'} features
        - Hidden Layer 1: {layer1_units} units (ReLU activation)
        - Hidden Layer 2: {layer2_units} units (ReLU activation)
        - Output Layer: {'1 unit (Linear)' if model_type == 'Regression' else 'Softmax activation'}
        """)
    
    with info_col2:
        st.info(f"""
        **Training Configuration:**
        - Optimizer: {optimizer_choice}
        - Loss Function: {'Mean Squared Error (MSE)' if model_type == 'Regression' else 'Sparse Categorical Crossentropy'}
        - Metrics: {'Mean Absolute Error (MAE)' if model_type == 'Regression' else 'Accuracy'}
        - Learning Rate: {learning_rate}
        """)
    
    if st.button("Train Model"):
        try:
            # Prepare data
            if model_type == "Regression":
                target_col = st.session_state.get('target_col', 'charges')
                X = processed_data[selected_features]
                y = processed_data[target_col]
            else:
                X = processed_data[selected_features]
                y = processed_data['charge_category_encoded']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler in session state
            st.session_state.scaler = scaler
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train baseline model
            status_text.text("Training baseline model...")
            progress_bar.progress(25)
            
            if model_type == "Regression":
                baseline_model = LinearRegression()
                baseline_model.fit(X_train_scaled, y_train)
                y_pred_baseline = baseline_model.predict(X_test_scaled)
                
                baseline_metrics = {
                    'MSE': mean_squared_error(y_test, y_pred_baseline),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_baseline)),
                    'MAE': mean_absolute_error(y_test, y_pred_baseline),
                    'R2': r2_score(y_test, y_pred_baseline)
                }
            else:
                baseline_model = LogisticRegression(max_iter=1000)
                baseline_model.fit(X_train_scaled, y_train)
                y_pred_baseline = baseline_model.predict(X_test_scaled)
                
                baseline_metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred_baseline)
                }
            
            # Train deep learning model
            status_text.text("Training deep learning model...")
            progress_bar.progress(50)
            
            # Select optimizer
            if optimizer_choice == "Adam":
                optimizer = Adam(learning_rate=learning_rate)
            else:
                optimizer = SGD(learning_rate=learning_rate)
            
            if model_type == "Regression":
                model = Sequential([
                    Dense(layer1_units, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                    Dense(layer2_units, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            else:
                n_classes = len(np.unique(y))
                model = Sequential([
                    Dense(layer1_units, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                    Dense(layer2_units, activation='relu'),
                    Dense(n_classes, activation='softmax')
                ])
                model.compile(optimizer=optimizer, 
                            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Train model
            status_text.text("Training in progress...")
            progress_bar.progress(75)
            
            history = model.fit(X_train_scaled, y_train, 
                              epochs=epochs, 
                              batch_size=batch_size, 
                              validation_split=validation_split, 
                              verbose=0)
            
            # Store model in session state
            st.session_state.model = model
            st.session_state.history = history
            st.session_state.model_type = model_type
            
            # Evaluate deep learning model
            status_text.text("Evaluating model...")
            progress_bar.progress(100)
            
            if model_type == "Regression":
                loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
                y_pred_dl = model.predict(X_test_scaled, verbose=0)
                
                dl_metrics = {
                    'MSE': loss,
                    'RMSE': np.sqrt(loss),
                    'MAE': mae,
                    'R2': r2_score(y_test, y_pred_dl)
                }
            else:
                loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                y_pred_dl = model.predict(X_test_scaled, verbose=0)
                y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)
                
                dl_metrics = {
                    'Accuracy': accuracy,
                    'Loss': loss
                }
            
            # Display results
            status_text.text("Training completed!")
            
            st.success("Model training completed!")
            
            # Show metrics comparison
            st.subheader("Model Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Baseline Model**")
                for metric, value in baseline_metrics.items():
                    st.metric(metric, f"{value:.4f}")
            
            with col2:
                st.write("**Deep Learning Model**")
                for metric, value in dl_metrics.items():
                    st.metric(metric, f"{value:.4f}")
            
            # Plot training history
            st.subheader("Training History")
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Metric'))
            
            # Loss plot
            fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'), row=1, col=1)
            fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss'), row=1, col=1)
            
            # Metric plot
            metric_key = 'mae' if model_type == "Regression" else 'accuracy'
            fig.add_trace(go.Scatter(y=history.history[metric_key], name=f'Train {metric_key}'), row=1, col=2)
            fig.add_trace(go.Scatter(y=history.history[f'val_{metric_key}'], name=f'Val {metric_key}'), row=1, col=2)
            
            fig.update_layout(height=400, title_text="Training History")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")

def create_prediction_section():
    """Create the prediction section"""
    st.markdown('<div class="section-header">🔮 Make Predictions</div>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first.")
        return
    
    model = st.session_state.model
    scaler = st.session_state.scaler
    selected_features = st.session_state.selected_features
    encoders = st.session_state.encoders
    model_type = st.session_state.model_type
    
    st.subheader("Input New Data for Prediction")
    
    # Create input form
    with st.form("prediction_form"):
        input_data = {}
        
        # Create input fields for each feature
        cols = st.columns(3)
        for i, feature in enumerate(selected_features):
            col_idx = i % 3
            
            with cols[col_idx]:
                if feature in encoders:
                    # Categorical feature
                    encoder = encoders[feature]
                    options = encoder.classes_
                    selected_value = st.selectbox(f"Select {feature}", options)
                    input_data[feature] = encoder.transform([selected_value])[0]
                else:
                    # Numerical feature
                    if feature == 'age':
                        input_data[feature] = st.number_input(f"{feature}", min_value=18, max_value=100, value=30)
                    elif feature == 'bmi':
                        input_data[feature] = st.number_input(f"{feature}", min_value=15.0, max_value=50.0, value=25.0)
                    elif feature == 'children':
                        input_data[feature] = st.number_input(f"{feature}", min_value=0, max_value=10, value=0)
                    else:
                        input_data[feature] = st.number_input(f"{feature}", value=0.0)
        
        predict_button = st.form_submit_button("Make Prediction")
    
    if predict_button:
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            if model_type == "Regression":
                # Predict cost directly
                predicted_cost = model.predict(input_scaled, verbose=0)[0][0]
                
                # Derive charge category from cost
                predicted_category, category_emoji = get_charge_category_from_cost(predicted_cost)
                
                # Display results
                st.success(f"**Predicted Insurance Cost: ${predicted_cost:.2f}**")
                st.success(f"**Predicted Charge Category: {category_emoji} {predicted_category}**")
                
                # Add detailed context
                cost_range = get_cost_range_from_category(predicted_category)
                st.info(f"""
                **Prediction Details:**
                - Predicted Cost: ${predicted_cost:.2f}
                - Charge Category: {predicted_category}
                - Category Range: ${cost_range[0]:,} - ${cost_range[1]:,}
                """)
                    
            else:
                # Predict category directly
                prediction_proba = model.predict(input_scaled, verbose=0)[0]
                predicted_class = np.argmax(prediction_proba)
                
                # Get category name
                charge_encoder = encoders['charge_category']
                predicted_category = charge_encoder.inverse_transform([predicted_class])[0]
                confidence = prediction_proba[predicted_class] * 100
                
                # Estimate cost from category
                cost_range = get_cost_range_from_category(predicted_category)
                estimated_cost = (cost_range[0] + cost_range[1]) / 2  # Use midpoint as estimate
                
                category_emoji = get_charge_category_from_cost(estimated_cost)[1]
                
                # Display results
                st.success(f"**Predicted Charge Category: {category_emoji} {predicted_category}**")
                st.success(f"**Estimated Insurance Cost: ${estimated_cost:.2f}**")
                
                st.info(f"""
                **Prediction Details:**
                - Predicted Category: {predicted_category} (Confidence: {confidence:.1f}%)
                - Estimated Cost: ${estimated_cost:.2f}
                - Category Range: ${cost_range[0]:,} - ${cost_range[1]:,}
                """)
                
                # Show all probabilities
                st.subheader("Category Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Category': charge_encoder.inverse_transform(range(len(prediction_proba))),
                    'Probability': prediction_proba * 100
                })
                prob_df = prob_df.sort_values('Probability', ascending=False)
                
                fig = px.bar(prob_df, x='Category', y='Probability', 
                           title='Prediction Probabilities by Category (%)',
                           labels={'Probability': 'Probability (%)'})
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🏥 Medical Insurance Cost Prediction Dashboard</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard provides comprehensive analysis and prediction capabilities for medical insurance costs.
    Upload your own dataset or use the default insurance dataset to explore data, train models, and make predictions.
    """)
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    
    # Data upload section
    st.sidebar.subheader("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    use_default = st.sidebar.button("Use Default Dataset")
    
    # Load data
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")
            st.sidebar.write(f"Shape: {data.shape}")
        except Exception as e:
            st.sidebar.error(f"Error uploading file: {str(e)}")
            data = None
    elif use_default:
        data = load_default_data()
        if data is not None:
            st.session_state.data = data
            st.sidebar.success("Default dataset loaded")
            st.sidebar.write(f"Shape: {data.shape}")
    else:
        data = st.session_state.data
    
    # Main sections
    if data is not None:
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🔧 Preprocessing", "🤖 Model Training", "🔮 Predictions"])
        
        with tab1:
            create_eda_section(data)
        
        with tab2:
            processed_data, selected_features = create_preprocessing_section(data)
        
        with tab3:
            processed_data = st.session_state.get('processed_data')
            selected_features = st.session_state.get('selected_features')
            create_model_training_section(processed_data, selected_features)
        
        with tab4:
            create_prediction_section()
    
    else:
        st.info("👆 Please upload a CSV file or use the default dataset to get started.")
        
        # Show sample data format
        st.subheader("Expected Data Format")
        st.markdown("""
        Your CSV file should contain columns similar to the medical insurance dataset:
        - **age**: Age of the individual
        - **sex**: Gender (male/female)
        - **bmi**: Body Mass Index
        - **children**: Number of children
        - **smoker**: Smoking status (yes/no)
        - **region**: Geographic region
        - **charges**: Medical insurance charges (target variable)
        """)
        
        # Sample data preview
        sample_data = pd.DataFrame({
            'age': [19, 18, 28, 33, 32],
            'sex': ['female', 'male', 'male', 'male', 'male'],
            'bmi': [27.9, 33.77, 33.0, 22.705, 28.88],
            'children': [0, 1, 3, 0, 0],
            'smoker': ['yes', 'no', 'no', 'no', 'no'],
            'region': ['southwest', 'southeast', 'southeast', 'northwest', 'northwest'],
            'charges': [16884.92, 1725.55, 4449.46, 21984.47, 3866.86]
        })
        
        st.subheader("Sample Data")
        st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()