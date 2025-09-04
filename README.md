# 🏡 House Price Prediction (KNN & Linear Regression)

This project applies **K-Nearest Neighbors (KNN)** and **Linear Regression** models to predict house prices.  
It includes data cleaning, preprocessing, model training, and evaluation using metrics such as **MAPE, RMSE, and Log RMSE**.

---

## 📊 Project Overview
- **Goal**: Predict house prices using regression algorithms.
- **Models Used**:
  - K-Nearest Neighbors (KNN)
  - Linear Regression
- **Evaluation Metrics**:
  - **MAPE** (Mean Absolute Percentage Error) → shows prediction error in percentage form (relative to true price).
  - **MAE** (Mean Absolute Error) → shows the average absolute error in raw units (but less interpretable).
  - **RMSE** (Root Mean Squared Error) → penalizes larger errors more heavily.
  - **Log RMSE** → preferred in house price prediction competitions (e.g., Kaggle) to handle skewed price distributions.

## ⚙️ How to Run the Project

Clone the repository:
```bash
git clone https://github.com/Akchiche-Mohamed-Aymen/ai-house-price-prediction-compare-algorithms-.git
```
Go to project folder
```
cd ai-house-price-prediction-compare-algorithms
````

### Step 1: Prepare Training Data

Clean and preprocess the training dataset:

```bash
py prepare_data.py
```

### Step 2: Prepare Test Data

Clean and preprocess the test dataset:

```bash
py clean_test.py
```

### Step 3: Run Linear Regression Model

Train and evaluate the Linear Regression model:

```bash
py linear_regression.py
```

### Step 4: Run KNN Model

Train and evaluate the KNN model:

```bash
py knn.py
```
## 🛠️ Tech Stack

* **Python 3**
* **Pandas** (data handling)
* **NumPy** (numerical computations)
* **Scikit-learn** (ML models and metrics)

---

## 📌 Notes

* **Why Log RMSE?**
  House prices are highly skewed (many cheap houses, few very expensive ones).
  Log RMSE penalizes relative errors, making it fairer across price ranges.
* **Why MAPE?**
  Unlike MAE, MAPE tells how far predictions are in *percentage terms* → more interpretable.
