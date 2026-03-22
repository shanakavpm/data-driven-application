# 💳 Data-Driven Credit Card Fraud Detection

This application is designed to detect fraudulent credit card transactions using machine learning models. It includes a complete pipeline for data preprocessing, exploratory data analysis (EDA), model training, and a web interface for real-time predictions.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed. You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Running the Application

1. **Train Model:** Run the training script to build and save your model.
   ```bash
   python train.py
   ```

2. **Web Interface:** Start the web application to interact with your data and predictions.
   ```bash
   python run.py
   ```

## 📂 Project Structure

- `src/`: Core logic for data handling, preprocessing, and model building.
- `webapp/`: Flask-based web interface (templates and static files).
- `models/`: Trained machine learning models.
- `plots/`: Visualizations generated during exploratory data analysis.
- `data/`: CSV datasets used for training and evaluation.

## 🛠 Features

*   **Exploratory Data Analysis (EDA):** Insights into transaction characteristics and distribution.
*   **Machine Learning Models:** Trained algorithms optimized for fraud detection.
*   **Real-time Prediction:** User-friendly web interface for individual transaction assessments.
*   **Automated Pipeline:** Clean separation of concerns with dedicated modules for each stage.

---
*Created for the Data-Driven Application assignment.*
