# Stock Price Prediction Using Machine Learning

This repository contains a machine learning-driven stock price prediction project in Python, developed to analyze and forecast stock prices based on historical data. It leverages multiple machine learning models, including Random Forest, LightGBM, CatBoost, and LSTM, and summarizes their performance in a custom table with metrics such as accuracy, precision, recall, and F1 score to help identify the optimal model for each stock.

## Project Overview
The goal of this project is to provide a data-driven approach to stock price prediction by:
1. **Data Analysis**: Analyzing historical stock price data to uncover patterns and trends.
2. **Model Implementation**: Implementing and fine-tuning various machine learning and deep learning models for prediction.
3. **Performance Comparison**: Creating a tabular summary that evaluates model performance across different metrics, highlighting the most suitable algorithm for each stock.

## Key Features
- **Data Preprocessing**: Cleanses, normalizes, and transforms data for improved model accuracy.
- **Feature Engineering**: Constructs technical indicators, such as moving averages, RSI, and MACD, to provide the model with additional context.
- **Machine Learning Models**: Applies a variety of algorithms, including Random Forest, LightGBM, CatBoost, and LSTM, to analyze different predictive techniques.
- **Custom Performance Summary**: A performance table displaying accuracy, precision, recall, F1 score, and directional prediction, showcasing each model's strengths.

## Technology Stack
- **Programming Language**: Python
- **Data Processing**: Pandas, NumPy
- **Machine Learning Libraries**: Scikit-Learn, LightGBM, CatBoost, TensorFlow/Keras (for LSTM)
- **Visualization**: Matplotlib, Seaborn


## Models and Evaluation

### Models Used
The project applies the following models to provide robust prediction capabilities:
- **Random Forest**: An ensemble method that constructs multiple decision trees to capture complex patterns in the stock data.
- **LightGBM**: A high-performance, gradient-boosting framework optimized for speed and efficiency, especially with large datasets.
- **CatBoost**: A gradient boosting algorithm optimized for handling categorical data and providing fast training.
- **LSTM (Long Short-Term Memory)**: A deep learning model designed to capture temporal dependencies in time series data, well-suited for stock price prediction.

### Performance Evaluation and Summary Table
Each model is evaluated on the following metrics:
- **Accuracy**: Measures the overall correctness of the model's predictions.
- **Precision**: Indicates the proportion of positive predictions that were correct.
- **Recall**: Shows the model's ability to identify all relevant instances.
- **F1 Score**: Combines precision and recall into a single metric to evaluate balanced performance.
- **Directional Prediction**: Evaluates whether the model correctly predicts the direction of price movement (e.g., increase or decrease).

#### Example Performance Table:
| Stock   | Accuracy | Precision | Recall | F1 Score | Direction | Best Algorithm |
|---------|----------|-----------|--------|----------|-----------|----------------|
| AEFES   | 0.60     | 0.61      | 0.68   | 0.64     | -1        | CatBoost       |
| ASELS   | 0.72     | 0.75      | 0.70   | 0.72     | 1         | LightGBM       |
| GARAN   | 0.65     | 0.63      | 0.67   | 0.65     | 1         | Random Forest  |
| THYAO   | 0.70     | 0.72      | 0.69   | 0.70     | -1        | LSTM           |

The table helps identify the best algorithm for each stock by comparing how well each model performs across these metrics.
