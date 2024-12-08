

# **Housing Prices Prediction**

This project is a machine learning pipeline for predicting housing prices using cleaned and processed datasets. The pipeline involves data preprocessing, polynomial regression modeling, and performance evaluation.  

## **Project Overview**

The goal is to predict house prices using the provided training and testing datasets. This involves:
- Cleaning and preprocessing the data.
- Building a polynomial regression model.
- Generating predictions for the test dataset.

## **Features**
- Data Cleaning: Handles missing values, removes outliers, and transforms features for better model performance.
- Polynomial Regression: Incorporates feature interactions and squared terms for improved predictions.
- Performance Evaluation: Analyzes model performance using residual plots, scatter plots, and R² metrics.

---

## **Table of Contents**
1. [Installation](#installation)
2. [Usage](#usage)
3. [File Structure](#file-structure)
4. [Code Explanation](#code-explanation)
5. [Results](#results)
6. [Contributing](#contributing)
7. [Acknowledgments](#acknowledgments)

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/mourvijoshi/PRODIGY_ML_01.git
   ```
2. Navigate to the project directory:
   ```bash
   cd PRODIGY_ML_01
   ```
3. Install the required Python libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

---

## **Usage**

1. **Prepare the Datasets**:
   - Place the `train.csv` and `test.csv` files in the project directory.

2. **Run Data Cleaning**:
   Execute the data cleaning script to preprocess the datasets:
   ```bash
   python data_cleaning.py
   ```
   Output:
   - `train_cleaned.csv`
   - `test_cleaned.csv`

3. **Run the Prediction Pipeline**:
   Use the machine learning pipeline to train the model and generate predictions:
   ```bash
   python prediction_task.py
   ```
   Output:
   - `sample_submission.csv`: Contains the predicted house prices for the test dataset.

---

## **File Structure**
```
PRODIGY_ML_01/
│
├── data_cleaning.py            # Data preprocessing and cleaning
├── task1.py          # Polynomial regression and predictions
├── train.csv           # Cleaned training dataset (output of data_cleaning.py)
├── test.csv            # Cleaned testing dataset (output of data_cleaning.py)
├── sample_submission.csv       # Predicted prices (output of prediction_task.py)
├── README.md                   # Project documentation
```

---

## **Code Explanation**

### **1. Data Cleaning**
- Handles missing values:
  - Numerical columns: Replaces with median values.
  - Categorical columns: Replaces with `'None'`.
- Removes outliers in `LotArea` using the IQR method.
- Applies log-transformation on `SalePrice` to reduce skewness.
- One-hot encodes categorical variables for model compatibility.
- Aligns training and testing datasets to ensure consistent features.

### **2. Prediction Task**
- Selects key features for training and testing.
- Uses polynomial regression to incorporate feature interactions.
- Standardizes features using `StandardScaler`.
- Evaluates model performance on training data using:
  - Mean Squared Error (MSE)
  - R-squared (R²) Score
- Saves test predictions in `sample_submission.csv`.

---

## **Results**
- **Training Performance**:
  - R-squared Score: Demonstrates model's ability to explain variance in training data.
  - Residual analysis: Checks for unbiased prediction errors.
- **Test Predictions**:
  - House prices for test data are saved in `sample_submission.csv`.

---

## **Contributing**
Feel free to fork this repository and make contributions. Pull requests are welcome!

---

## **Acknowledgments**
This project was created as part of a machine learning task to predict house prices. Special thanks to the open-source community for tools and libraries like pandas, scikit-learn, and matplotlib.

