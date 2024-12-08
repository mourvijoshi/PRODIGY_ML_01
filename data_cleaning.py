import pandas as pd
import numpy as np

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


numerical_cols = ['LotFrontage','MasVnrArea','GarageYrBlt']
categorical_cols = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

for col in numerical_cols:
    df_train[col] = df_train[col].fillna(df_train[col].median())
    df_test[col] = df_test[col].fillna(df_test[col].median())


for col in categorical_cols:
    df_train[col] = df_train[col].fillna('None')
    df_test[col] = df_test[col].fillna('None')

# Remove outliers from LotArea
Q1 = df_train['LotArea'].quantile(0.25)
Q3 = df_train['LotArea'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_train = df_train[(df_train['LotArea'] >= lower_bound) & (df_train['LotArea'] <= upper_bound)]

# Log-transform SalePrice to handle skewness
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

# One-hot encode categorical variables
df_train = pd.get_dummies(df_train, drop_first=True)
df_test = pd.get_dummies(df_test, drop_first=True)

# Align train and test datasets
df_train, df_test = df_train.align(df_test, join='left', axis=1)

# Save cleaned datasets
df_train.to_csv('train_cleaned.csv', index=False)
df_test.to_csv('test_cleaned.csv', index=False)

print("Data cleaning completed. Cleaned datasets saved as 'train_cleaned.csv' and 'test_cleaned.csv'.")
