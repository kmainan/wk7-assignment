# Task 1: Load and Explore the Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])
    print("Dataset loaded successfully.")
except Exception as e:
    print("Error loading dataset:", e)

# Display first few rows
print("First 5 rows:")
print(df.head())

# Check data types and missing values
print("\nData types and missing values:")
print(df.info())
print(df.isnull().sum())

# Clean the dataset (drop missing values if any)
df_cleaned = df.dropna()

# Task 2: Basic Data Analysis

# Compute basic statistics
print("\nBasic statistics:")
print(df_cleaned.describe())

# Group by species and compute the mean of numeric columns
grouped_means = df_cleaned.groupby('species').mean()
print("\nGrouped mean by species:")
print(grouped_means)

# Task 3: Data Visualization
sns.set(style="whitegrid")

# Line Chart: Cumulative Petal Length over index
df_cleaned['Cumulative Petal Length'] = df_cleaned['petal length (cm)'].cumsum()
plt.figure(figsize=(10, 5))
sns.lineplot(x=df_cleaned.index, y='Cumulative Petal Length', data=df_cleaned)
plt.title('Cumulative Petal Length Over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Cumulative Petal Length (cm)')
plt.tight_layout()
plt.show()

# Bar Chart: Average Petal Length by Species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df_cleaned, ci=None)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# Histogram: Distribution of Sepal Width
plt.figure(figsize=(8, 5))
sns.histplot(df_cleaned['sepal width (cm)'], bins=15, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df_cleaned)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()
