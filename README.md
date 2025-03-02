# Multiple Feature Linear Regression (Vectorized Implementation)

## 📌 Overview
This project implements a **Multiple Feature Linear Regression Model from scratch** using only **NumPy**. The model is fully **vectorized** to enhance computational efficiency and leverage **parallel hardware acceleration**.

## 🚀 Features
- **Built from Scratch**: No external ML libraries, only **NumPy**.
- **Vectorized Implementation**: Optimized using **matrix operations** for speed.
- **Scalable**: Can handle multiple features efficiently.
- **Prediction Task**: Predicts the **closing price of the next day** when today's values are given.

## 📊 Why NumPy?
Using **NumPy** allows us to:
- **Vectorize computations** for faster execution.
- **Leverage parallel processing** on modern CPUs and GPUs.
- **Efficiently handle large datasets** with matrix operations.

## 🏗️ Model Implementation
The model follows these steps:
1. **Data Preprocessing**: Normalization and handling missing values.
2. **Gradient Descent Optimization**: Used to minimize the cost function efficiently.
3. **Cost Function**: Mean Squared Error (MSE) to evaluate performance.
4. **Model Training**: Iterative weight updates using **gradient descent**.
5. **Prediction**: Given today's market data, predicts **tomorrow’s closing price**.

## 📈 Results & Performance
- **R² Score**: `0.9966` (indicating a strong fit, but requires further analysis for overfitting).
- **Evaluation Metrics**: Mean Squared Error (MSE), Residual Plots.

## 🔧 How to Use
1. Install NumPy: `pip install numpy`
2. Load dataset and preprocess features.
3. Train the model using the provided **vectorized gradient descent**.
4. Make predictions using `predict(X_test)`.

## 🛠️ Future Improvements
- Implement **regularization (L1/L2)** to avoid overfitting.
- Support **mini-batch gradient descent** for large-scale datasets.
- Extend to **polynomial regression** for non-linear relationships.

