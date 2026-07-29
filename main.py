import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from processing.classification import binomial_logistic_regression as BLR

# 1. Caricamento Dati
df = pd.read_excel(
    "data/CA_normo.xlsx",
    sheet_name="sw1000",
    header=0
)

# 2. Separazione Feature e Classe
data = df.iloc[:, 2:-1].to_numpy(dtype=float)
binary_class = df.iloc[:, -1].to_numpy(dtype=int)

# Codifica della stimolazione
type_of_stimulation = df.iloc[:, 1]
stimulation = (type_of_stimulation == "condensazione").astype(float)

# Unione della variabile stimulation con le altre feature
X = np.column_stack((stimulation, data))
y = binary_class

# 3. Train/Test Split Stratificato
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

# Ricomposizione per la funzione BLR (Feature + Target come ultima colonna)
training_set = np.column_stack((X_train, y_train))
test_set = np.column_stack((X_test, y_test))

# 4. Esecuzione Modello
k = 5
Beta, Accuracy_LR, Sensibility_LR, Specificity_LR = BLR(training_set, test_set, k)

# 5. Stampa dei Risultati
print("--- Risultati Regressione Logistica ---")
print("Coefficienti Beta ottimali:\n", Beta)
print(f"\nAccuracy Train (soglia 0.5): {Accuracy_LR[0][50]:.3f}")
print(f"Accuracy Test (soglia 0.5):  {Accuracy_LR[1][50]:.3f}")
print(f"Sensibilità Test (soglia 0.5): {Sensibility_LR[1][50]:.3f}")
print(f"Specificità Test (soglia 0.5): {Specificity_LR[1][50]:.3f}")