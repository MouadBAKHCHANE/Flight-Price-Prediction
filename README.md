# ✈️ Flight Price Prediction Dashboard

![Flight Price Prediction](airfare.png)

## 🚀 Live Demo
**[Click here to view the deployed app](https://mouadbakhchane-flight-price-prediction-app-55qrhn.streamlit.app/)**

## 📖 Project Overview
This project is a comprehensive data science and machine learning application designed to analyze and predict flight prices in the Indian Aviation Market. It leverages a robust dataset to provide interactive insights through an Exploratory Data Analysis (EDA) dashboard and estimates flight fares using a trained Gradient Boosting Regressor model.

The application is built with **Streamlit** for a seamless and responsive user interface, making complex data accessible and actionable.

## ✨ Key Features

### 📊 Interactive Dashboard
Navigate to the **Dashboard** tab to explore dynamic visualizations:
*   **Market Share Analysis**: Visualize the dominance of different airlines.
*   **Price Analysis**: Compare average ticket prices across airlines, stops, and service classes.
*   **Duration vs. Price**: Understand how flight duration impacts cost.
*   **Filter Data**: Use the sidebar filters to drill down by City, Airline, Service Class, Stops, and Time.

### 💸 Intelligent Price Prediction
Switch to the **Predict** tab to get real-time fare estimates:
*   **Input Details**: Select your Airline, Source/Destination Cities, Travel Date, and Time.
*   **Smart Features**: The app automatically calculates "Days Left" and categorizes departure times (e.g., Morning, Evening) to feed into the model.
*   **Accurate Estimates**: Powered by a Gradient Boosting Machine (GBM) model trained on historical data.

## 🛠️ Technology Stack
*   **Python**: Core programming language.
*   **Streamlit**: Web application framework.
*   **Pandas & NumPy**: Data manipulation and numerical operations.
*   **Plotly**: Interactive charting and visualization.
*   **Scikit-Learn**: Machine Learning implementation (Gradient Boosting).
*   **Joblib**: Model persistence.

## 📦 Installation & Local Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/MouadBAKHCHANE/Flight-Price-Prediction.git
    cd Flight-Price-Prediction
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**:
    ```bash
    streamlit run app.py
    ```

## 👨‍💻 Author
**Mouad Bakhchane**
*   [Website](http://www.mouadbakhchane.com)
*   [LinkedIn](https://www.linkedin.com/in/mouad-bakhchane)
*   [GitHub](https://github.com/MouadBAKHCHANE)
