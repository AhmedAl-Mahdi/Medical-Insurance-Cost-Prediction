# Medical Insurance Cost Prediction

A comprehensive machine learning project that performs exploratory data analysis (EDA) and builds deep learning models to predict medical insurance costs. This project includes both regression and classification approaches with an interactive Streamlit dashboard for real-time predictions.

## 🎯 Project Overview

This project analyzes medical insurance data to understand the factors that influence insurance costs and provides accurate cost predictions using advanced machine learning techniques. The analysis includes comprehensive EDA, data preprocessing, feature engineering, and multiple deep learning models for both regression and classification tasks.

## ✨ Features

- **📊 Exploratory Data Analysis (EDA)**: Comprehensive statistical analysis and visualization of medical insurance data
- **🔧 Data Preprocessing**: Advanced data cleaning, feature engineering, and standardization techniques
- **🧠 Deep Learning Models**: 
  - Neural network regression for continuous cost prediction
  - Classification models for cost category prediction
- **📈 Interactive Controls**: Real-time parameter adjustment and model comparison
- **📁 CSV Upload**: Easy data upload functionality for custom datasets
- **🔮 Prediction Engine**: Real-time insurance cost predictions with confidence intervals
- **📊 Model Comparison**: Side-by-side comparison of different algorithms
- **🎛️ Hyperparameter Tuning**: Automated optimization with cross-validation
- **📉 Convergence Analysis**: Training visualization and performance monitoring

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AhmedAl-Mahdi/Medical-Insurance-Cost-Prediction.git
   cd Medical-Insurance-Cost-Prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available, install the main dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn tensorflow matplotlib seaborn plotly
   ```

### Running the Streamlit Dashboard

1. **Start the Streamlit application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - The application will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

## 📖 How to Use the Dashboard

### 1. **Data Upload**
   - Use the sidebar to upload your own CSV file with medical insurance data
   - Or use the provided sample dataset for testing
   - Supported format: CSV files with columns (age, sex, bmi, children, smoker, region, charges)

### 2. **Exploratory Data Analysis**
   - Navigate to the "EDA" section to explore data distributions
   - View correlation matrices and statistical summaries
   - Analyze relationships between variables with interactive plots

### 3. **Model Training**
   - Select from multiple machine learning algorithms
   - Adjust hyperparameters using interactive controls
   - Monitor training progress with real-time visualizations

### 4. **Making Predictions**
   - Input individual patient information:
     - Age: Patient's age (18-100)
     - Sex: Male/Female
     - BMI: Body Mass Index (15-50)
     - Children: Number of dependents (0-10)
     - Smoker: Yes/No
     - Region: Geographic region
   - Get instant cost predictions with confidence intervals
   - View feature importance and prediction explanations

### 5. **Model Comparison**
   - Compare performance metrics across different models
   - View accuracy, MAE, RMSE, and R² scores
   - Analyze residual plots and prediction distributions

## 🛠️ Technical Details

### Dataset
The project uses a medical insurance dataset containing:
- **Age**: Age of the primary beneficiary
- **Sex**: Gender of the beneficiary
- **BMI**: Body mass index
- **Children**: Number of dependents
- **Smoker**: Smoking status
- **Region**: Geographic region
- **Charges**: Individual medical costs (target variable)

### Models Implemented
- **Deep Neural Networks**: Multi-layer perceptrons for regression
- **Classification Models**: Cost category prediction
- **Ensemble Methods**: Random Forest, Gradient Boosting
- **Traditional ML**: Linear Regression, SVM

### Performance Metrics
- **Regression**: RMSE, MAE, R², MAPE
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Cross-validation**: K-fold validation for robust evaluation

## 🔗 Links

- **GitHub Repository**: [https://github.com/AhmedAl-Mahdi/Medical-Insurance-Cost-Prediction](https://github.com/AhmedAl-Mahdi/Medical-Insurance-Cost-Prediction)
- **Streamlit App**: *[To be updated when deployed]*

## 📄 Files Structure

```
Medical-Insurance-Cost-Prediction/
├── README.md                 # Project documentation
├── Untitled0.ipynb         # Jupyter notebook with analysis
├── insurance.csv            # Dataset
├── app.py                   # Streamlit application (when implemented)
├── requirements.txt         # Python dependencies (when added)
└── models/                  # Saved model files (when implemented)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact & Citation

- **Author**: Ahmed Al-Mahdi
- **Email**: *[Contact information to be added]*
- **LinkedIn**: *[Profile link to be added]*

### Citation
If you use this project in your research or work, please cite:
```
Al-Mahdi, A. (2024). Medical Insurance Cost Prediction using Deep Learning. 
GitHub repository: https://github.com/AhmedAl-Mahdi/Medical-Insurance-Cost-Prediction
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: Medical Cost Personal Datasets
- TensorFlow and Keras teams for deep learning frameworks
- Streamlit team for the amazing web app framework
- Scikit-learn community for machine learning tools

---

*This project demonstrates the application of machine learning and deep learning techniques to real-world healthcare cost prediction problems, providing both educational value and practical utility for insurance cost estimation.*  
