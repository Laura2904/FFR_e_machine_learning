#Main for Auditory Capacity index assessment
import pandas as pd
import numpy as np

#Data loading
df = pd.read_excel("data/CA_normo.xlsx",sheet_name="sw1000",header=0)

# Convert binary categorical columns to 0/1
for col in df.select_dtypes(include=['object', 'category']):
    valori = df[col].dropna().unique()

    if len(valori) == 2:
        df[col] = df[col].map({
            valori[0]: 0,
            valori[1]: 1
        })

# Feature and class separation
data = df.iloc[:, 2:-1].to_numpy(dtype=float)
binary_class = df.iloc[:, -1].to_numpy(dtype=int)

features_names=df.columns[2:-1].to_list()

#Z-score
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=1)
std[std == 0] = 1


