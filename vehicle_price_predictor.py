# -*- coding: utf-8 -*-
"""Vehicle-Price-Predictor.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1l4dFhjA-stVzsZtE8ufIAIm6urDk-7lE
"""

!pip install numpy -q
!pip install pandas -q
!pip install matplotlib -q
!pip install tensorflow -q

!pip install opendatasets -q

"""# **Importing Libraries**"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import opendatasets as od
import seaborn as sns
import sklearn

import os

data_url = "https://drive.google.com/uc?export=download&id=1MIfp-SDqb6jbR9fX8mZ8aKg4TI7ighgQ"
df = pd.read_csv(data_url)
df.head()

"""# **Dataset Description**"""

df.shape

df.describe()

df.shape

"""# **Exploratory Data Analysis**"""

print(df.columns)

df.dtypes
#types of data

eda_columns = df.select_dtypes(include=['int64','float64']).columns

eda_data = df[eda_columns]

# Generate summary statistics
summary_stats = eda_data.describe()

correlation_matrix = eda_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Significant Variables')
plt.show()

plt.figure(figsize=(18, 8))

for i, column in enumerate(eda_columns, 1):
    plt.subplot(1, len(eda_columns), i)


    if pd.api.types.is_numeric_dtype(df[column]):
        sns.boxplot(y=df[column], flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 8})
    else:
        print(f"Column {column} is not numeric. Skipping box plot.")

    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.grid(axis='y')

plt.tight_layout()
plt.show()

df_outliers = df

print(df_outliers.columns)

columns = ['year', 'price', 'mileage', 'tax' , 'mpg','engineSize']
outlier_counts = {}

for col in columns:
    z_scores = np.abs((df_outliers[col] - df_outliers[col].mean()) / df_outliers[col].std())
    outlier_counts[col] = (z_scores > 3).sum()

print("Outlier counts using Z-score:", outlier_counts)

# Identify and remove outliers using the IQR method
def remove_outliers_iqr(df, columns):
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Filter the data
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Example usage
numeric_columns = ['year', 'price', 'mileage', 'tax' , 'mpg','engineSize']
df_cleaned = remove_outliers_iqr(df_outliers, numeric_columns)

plt.figure(figsize=(18, 8))

for i, column in enumerate(eda_columns, 1):
    plt.subplot(1, len(eda_columns), i)


    if pd.api.types.is_numeric_dtype(df_cleaned[column]):
        sns.boxplot(y=df_cleaned[column], flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 8})
    else:
        print(f"Column {column} is not numeric. Skipping box plot.")

    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.grid(axis='y')

plt.tight_layout()
plt.show()

df_cleaned.shape

"""# **Data Preprocessing**"""

updated_df = df_cleaned

#checking null values
updated_df.isna().sum()

#checking null values
updated_df.isna().sum()

updated_df.shape

#duplicate values
updated_df.duplicated().sum()

updated_df = updated_df.drop_duplicates()

updated_df.shape

updated_df.dtypes

# Selecting columns with object datatypes
string_columns = updated_df.select_dtypes(include=['object'])
string_columns.head()

updated_df['brand'].unique()

updated_df['brand'].value_counts()

make_type_variables = pd.get_dummies(updated_df['brand'], dtype = int)
make_type_variables.head()

updated_df = pd.concat([updated_df, make_type_variables], axis = 1)
updated_df.head()

updated_df.shape

updated_df = updated_df.drop('brand', axis=1)
updated_df.head()

updated_df['model'].value_counts()

threshold = 800

# Identify rare models
model_counts = updated_df['model'].value_counts()
rare_models = model_counts[model_counts < threshold].index

# Replace with 'Other'
updated_df['model'] = updated_df['model'].replace(rare_models, 'other')

updated_df['model'].value_counts()

model_type_variables = pd.get_dummies(updated_df['model'], dtype = int)
model_type_variables.head()

updated_df = pd.concat([updated_df, model_type_variables], axis = 1)
updated_df.head()

updated_df = updated_df.drop('model', axis=1)
updated_df.head()

updated_df['transmission'].value_counts()

updated_df = updated_df[updated_df['transmission'] != 'Other']

updated_df['transmission'].value_counts()

transmission_type_variables = pd.get_dummies(updated_df['transmission'], dtype = int)
transmission_type_variables.head()

updated_df = pd.concat([updated_df, transmission_type_variables], axis = 1)
updated_df.head()

updated_df = updated_df.drop('transmission', axis=1)
updated_df.head()

updated_df['fuelType'].value_counts()

updated_df = updated_df[updated_df['fuelType'] != 'Other']
updated_df = updated_df[updated_df['fuelType'] != 'Hybrid']

updated_df['fuelType'].value_counts()

type_variables = pd.get_dummies(updated_df['fuelType'], dtype = int)
type_variables.head()

updated_df = pd.concat([updated_df, type_variables], axis = 1)
updated_df.head()

updated_df = updated_df.drop('fuelType', axis=1)
updated_df.head()

data = updated_df

data.head()

"""# **Model Implementation**"""

X = data.drop('price', axis=1)
y = data['price']

print(X.columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

print(X_train.dtypes)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(1024),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Dense(512),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),


    tf.keras.layers.Dense(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(
    loss=tf.keras.losses.mae,  # Mean Absolute Error
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['mae'],

)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.01 * 0.95 ** epoch)

epoch_number = 100
# Train the model
history = model.fit(X_train, y_train,
          epochs=epoch_number,
          batch_size=32,
          validation_data=(X_test, y_test),
          callbacks=[lr_schedule])

model.summary()

y_prediction = model.predict(X_test)
y_prediction[:5]

y_test.head()

"""# **Evaluation Metrics**"""

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_test.shape, y_prediction.shape)

y_prediction = y_prediction.ravel()

plt.scatter(X_train.iloc[:, 0], y_train, color='blue', label='Training Data')
plt.scatter(X_test.iloc[:, 0], y_test, color='red', label='Testing Data')
plt.scatter(X_test.iloc[:, 0], y_prediction, color='green', label='Predicted Data')
plt.legend()
plt.show()

mae_metric = tf.keras.losses.MeanAbsoluteError()

# Calculate MAE
mae_value = mae_metric(y_test, y_prediction).numpy()

print(f"Mean Absolute Error: {mae_value}")

mse_metric = tf.keras.losses.MeanSquaredError()


mse_value = mse_metric(y_test, y_prediction).numpy()

print(f"Mean Squared Error: {mse_value}")

x_range = range(1, epoch_number+1)
loss= history.history['loss']
plt.plot(x_range, loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()

train_score = model.evaluate(X_train, y_train, verbose=0)[1]

# Evaluate model on testing data
test_score = model.evaluate(X_test, y_test, verbose=0)[1]

print(f"Model Performance on Testing Dataset (MAE): {test_score:.4f}")

from sklearn.metrics import r2_score

# Predict on training data
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)

# Predict on testing data
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Model Accuracy (R²) on Training Dataset: {train_r2 * 100:.4f}%")
print(f"Model Accuracy (R²) on Testing Dataset: {test_r2 * 100:.4f}%")

import pickle as pk

pk.dump(model,open('PriceModel.pkl','wb'))

