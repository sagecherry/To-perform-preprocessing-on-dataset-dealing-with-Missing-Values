# To-perform-preprocessing-on-dataset-dealing-with-Missing-Values
# THEORY:
This experiment demonstrates basic **data preprocessing** and **handling of missing values** using Pandas in Python. Real-world datasets often contain missing (NaN), inconsistent, or incorrect values that must be cleaned before analysis or modeling.
The notebook covers key steps:

1. Creating a small sample DataFrame with missing values  
   ```python
   import pandas as pd
   import numpy as np

   data = {'c1': [11, 21, np.nan],
           'c2': [15, np.nan, np.nan],
           'c3': [10, 20, 30]}
   df = pd.DataFrame(data)
   ```

2. Detecting missing values  
   ```python
   df.isna().sum()          # count NaN per column
   df.isnull().sum()        # same as isna()
   df.notna().sum()         # count non-missing values
   ```

3. Dropping rows or columns with missing values  
   ```python
   df.dropna()              # drop any row with at least one NaN
   df.dropna(axis=1)        # drop columns with any NaN
   df.dropna(thresh=2)      # keep rows with at least 2 non-NaN values
   ```

4. Filling missing values  
   - With a constant  
     ```python
     df.fillna(0)
     df['c1'].fillna(999)
     ```

   - Forward fill / backward fill  
     ```python
     df.fillna(method='ffill')   # propagate previous value forward
     df.fillna(method='bfill')   # propagate next value backward
     ```

   - With statistical measures (mean, median, mode)  
     ```python
     df['c1'].fillna(df['c1'].mean())
     df['c2'].fillna(df['c2'].median())
     df['c3'].fillna(df['c3'].mode()[0])
     ```

5. Handling missing values in a real student dataset (student_data.csv)  
   - Issues: missing age (-), missing marks (-), inconsistent date format, missing admission_date in cleaned version  
   - Steps shown:  
     - Replace invalid markers ('-') with NaN  
     - Convert age and marks to numeric (float)  
     - Fill missing age with mean/median  
     - Fill missing marks with mean or median  
     - Standardize date format (to YYYY-MM-DD)  
     - Save cleaned version → cleaned_student_data.csv

6. Handling missing values in the Cars93 dataset (Cars93 (1).csv)  
   - Missing values found in: AirBags (34), Rear.seat.room (2), Luggage.room (11)  
   - Steps shown:  
     - Check missing counts: df.isna().sum()  
     - Fill AirBags with mode (most frequent value: "Driver only")  
       ```python
       df['AirBags'] = df['AirBags'].fillna(df['AirBags'].mode()[0])
       ```
     - (Other columns like Rear.seat.room and Luggage.room can be filled with mean/median if numeric)  
     - Save cleaned version → cleaned_cars_data.csv

These techniques are essential for:
- Preparing data for machine learning (most algorithms cannot handle NaN)
- Improving data quality and reliability
- Avoiding bias introduced by improper imputation

## Conclusion:

The experiment successfully demonstrated how to:
- Identify missing values using isna()/isnull()
- Remove rows/columns with dropna()
- Impute missing data using constants, forward/backward fill, mean, median, or mode
- Clean a small toy dataset, a student records dataset, and the classic Cars93 dataset
- Standardize formats (e.g., dates) and save cleaned versions as new CSV files

Key observations:
- Simple statistical imputation (mean/median/mode) is common for numeric columns
- Mode is useful for categorical columns (e.g., AirBags filled with "Driver only")
- Always check data types and invalid entries ('-', empty strings) before numeric conversion
- After cleaning, missing value count should be zero (or intentionally kept if needed)
