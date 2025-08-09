# 🩺 Medical Insurance Cost Prediction

A comprehensive machine learning application for predicting medical insurance costs with a professional healthcare-focused user interface.

## Overview

This project provides an advanced Streamlit dashboard that performs exploratory data analysis, model training, and cost prediction for medical insurance charges. The application features a medical-themed interface designed specifically for healthcare analytics.

## Features

### 🔬 Core Analytics
- **Exploratory Data Analysis**: Interactive visualizations and statistical summaries
- **Model Training**: Support for Linear Regression, Random Forest, and Neural Networks
- **Cost Prediction**: Real-time insurance cost estimation with risk categorization
- **Performance Metrics**: Comprehensive model evaluation with RMSE, R², MAE, and MSE

### 🎨 Medical-Themed UI
- **Professional Healthcare Design**: Calming blue/teal color palette with medical iconography
- **Responsive Layout**: Mobile-friendly design with organized card-based sections
- **Interactive Elements**: Styled metric cards, category chips, and progress indicators
- **Clinical Context**: Educational disclaimers and professional guidance

### 🌙 Customization Options
- **Dark Mode Toggle**: Switch between light and dark themes
- **Clinical Context Display**: Optional safety disclaimers
- **Custom Fonts**: Modern Inter typography via Google Fonts integration

## Quick Start

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

**Basic installation (Linear Regression + Random Forest models):**
```bash
pip install -r requirements.txt
```

**Full installation (includes Neural Network models):**
```bash
pip install -r requirements-full.txt
```

> **Note**: The basic installation excludes TensorFlow to ensure compatibility with lightweight deployment environments. Neural Network models will be automatically disabled if TensorFlow is not available.

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Usage Guide

### 1. Exploratory Data Analysis 📊
- View dataset statistics and key metrics
- Explore data distributions and correlations
- Analyze feature relationships with interactive charts

### 2. Model Training 🧬
- Choose from multiple model architectures
- Configure training parameters
- Monitor performance metrics in real-time

### 3. Cost Prediction 💊
- Input patient demographics and characteristics
- Get instant cost predictions with confidence categories
- View risk classifications (Low/Medium/High/Very High)

## Styling & Theming

### Custom Theme Configuration
The application uses a medical-themed design defined in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#2E86AB"  # Medical blue
backgroundColor = "#F0F8FF"  # Light blue background
secondaryBackgroundColor = "#FFFFFF"  # White cards
textColor = "#2C3E50"  # Dark gray text
```

### Color Palette
- **Primary Blue**: `#2E86AB` - Main interactive elements
- **Primary Teal**: `#A23B72` - Accent and gradients
- **Success Green**: `#28A745` - Low-risk categories
- **Warning Orange**: `#FFC107` - Medium-risk categories
- **Alert Red**: `#F18F01` - High-risk categories

### Dark Mode
Toggle dark mode using the sidebar control. The application automatically adjusts:
- Background colors and gradients
- Text contrast ratios (WCAG AA compliant)
- Card and component styling
- Category chip colors

### Customizing the Logo
To add your own logo:
1. Place your logo file in the `assets/` directory
2. Update the header section in `app.py`:
```python
# Add your logo reference in create_medical_header()
st.image("assets/your_logo.png", width=200)
```

## Technical Architecture

### Dependencies
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning models and preprocessing
- **TensorFlow/Keras**: Neural network implementation
- **Pandas/NumPy**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts and graphs

### File Structure
```
Medical-Insurance-Cost-Prediction/
├── app.py                          # Main Streamlit application
├── .streamlit/
│   └── config.toml                 # Theme and server configuration
├── requirements.txt                # Python dependencies
├── insurance.csv                   # Dataset
├── README.md                       # Documentation
└── assets/                         # Optional logo and media files
```

### Key Functions
- `inject_global_styles()`: CSS injection for medical theming
- `create_medical_header()`: Professional header with branding
- `eda_section()`: Interactive data exploration
- `model_training_section()`: ML model training interface
- `prediction_section()`: Cost prediction with styled outputs

## Clinical Context & Disclaimers

⚠️ **Important**: This application is designed for educational and demonstration purposes only. Predictions should not be used for:
- Actual medical decision-making
- Real insurance pricing or underwriting
- Clinical diagnosis or treatment planning

Always consult with qualified healthcare and insurance professionals for real-world applications.

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Follow the existing code style and medical theming
4. Test thoroughly with the provided dataset
5. Submit a pull request with detailed description

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

*Built with ❤️ for the healthcare analytics community*
