# Medical Insurance Cost Prediction Dashboard 🏥

This project provides a comprehensive Streamlit dashboard for analyzing and predicting medical insurance costs using deep learning models. The dashboard includes exploratory data analysis, data preprocessing, model training, and interactive prediction capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone https://github.com/AhmedAl-Mahdi/Medical-Insurance-Cost-Prediction.git
cd Medical-Insurance-Cost-Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit dashboard:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📊 Dashboard Features

### 1. **Exploratory Data Analysis (EDA)**
- Dataset overview and summary statistics
- Interactive visualizations:
  - Distribution plots (histograms and box plots)
  - Relationship analysis (scatter plots)
  - Categorical analysis (bar charts and pie charts)
  - Correlation heatmaps

### 2. **Data Preprocessing**
- Feature and target selection
- Automatic categorical encoding
- Feature scaling
- Charge category creation for classification

### 3. **Model Training**
- **Regression Models**: Predict exact insurance charges
- **Classification Models**: Categorize charges (Low, Medium, High, Very High)
- **Hyperparameter Controls**:
  - Number of epochs (10-500)
  - Batch size options
  - Neural network architecture
  - Learning rate selection
- Real-time training progress and metrics visualization

### 4. **Interactive Predictions**
- Input forms for new data predictions
- Regression: Get predicted insurance charges with context
- Classification: Get charge category with confidence scores
- Probability distribution visualization

### 5. **Custom Data Upload**
- Upload your own CSV files
- Compatible with medical insurance data format
- Automatic data validation and preprocessing

## 📁 Dataset Information

The default dataset (`insurance.csv`) contains the following columns:
- **age**: Age of the individual
- **sex**: Gender (male/female)
- **bmi**: Body Mass Index
- **children**: Number of children/dependents
- **smoker**: Smoking status (yes/no)
- **region**: Geographic region (northeast, northwest, southeast, southwest)
- **charges**: Medical insurance charges (target variable)

## 🎯 Usage Instructions

1. **Load Data**: Use the default dataset or upload your own CSV file
2. **Explore Data**: Navigate to the EDA tab to understand your data
3. **Preprocess**: Configure preprocessing options and feature selection
4. **Train Models**: Set hyperparameters and train both regression and classification models
5. **Make Predictions**: Input new data to get predictions and insights

## 🔧 Technical Implementation

- **Frontend**: Streamlit with custom CSS styling
- **Machine Learning**: TensorFlow/Keras for deep learning models
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualizations**: Plotly for interactive charts
- **Models**: Neural networks for both regression and classification tasks

## 📈 Model Performance

The dashboard compares deep learning models against baseline models:
- **Regression**: Linear Regression vs Deep Neural Network
- **Classification**: Logistic Regression vs Deep Neural Network
- Real-time performance metrics and training visualization

## 🎨 Screenshots

![Dashboard Screenshot](https://github.com/user-attachments/assets/daf8a9bb-e1f3-4bfa-9743-9a1e18992928)

---

The goal of this project is to perform comprehensive exploratory data analysis and regression modeling on the medical cost personal dataset to predict insurance charges.
