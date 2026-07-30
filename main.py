import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from processing.classification_LR import binomial_logistic_regression as BLR

# Data loading
df = pd.read_excel(
    "data/CA_normo.xlsx",
    sheet_name="sw1000",
    header=0
)

# feature and class separation
data = df.iloc[:, 2:-1].to_numpy(dtype=float)
binary_class = df.iloc[:, -1].to_numpy(dtype=int)

# stimulation encoding
type_of_stimulation = df.iloc[:, 1]
stimulation = (type_of_stimulation == "condensazione").astype(int)

X = np.column_stack((stimulation, data))
y = binary_class

# Train/Test Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=20, 
    stratify=y
)

# union features and class
training_set = np.column_stack((X_train, y_train))
test_set = np.column_stack((X_test, y_test))

# training model
k = 5
Beta, Accuracy_LR, Sensibility_LR, Specificity_LR = BLR(training_set, test_set, k)

# Results
print("--- Logistic Regression results ---")
print("Beta coefficents:\n", Beta)
print(f"\nAccuracy Train (threshold 0.5): {Accuracy_LR[0][50]:.3f}")
print(f"Accuracy Test (threshold 0.5):  {Accuracy_LR[1][50]:.3f}")
print(f"Sensibility Test (threshold 0.5): {Sensibility_LR[1][50]:.3f}")
print(f"Specificity Test (threshold 0.5): {Specificity_LR[1][50]:.3f}")

#find the best threshold
Sensibility_train = Sensibility_LR[0]
Specificity_train = Specificity_LR[0]

youden_index =  (Sensibility_train + Specificity_train)-1
alpha = np.linspace(1, 0, 101)
idx_optimal_threshold = np.argmax(youden_index)
optimal_threshold = alpha[idx_optimal_threshold]
max_youden_val = youden_index[idx_optimal_threshold]

print(f"The best threshold(Alpha): {optimal_threshold:.3f}")
print(f"Best Youden value: {max_youden_val:.3f}")
print(f"Accuracy Test (the best threshold):  {Accuracy_LR[1][idx_optimal_threshold]:.3f}")
print(f"Sensibility Test (the best threshold): {Sensibility_LR[1][idx_optimal_threshold]:.3f}")
print(f"Specificity Test (the best threshold): {Specificity_LR[1][idx_optimal_threshold]:.3f}")


#ROC
sens_test = Sensibility_LR[1]   # True Positive Rate (TPR)
spec_test = Specificity_LR[1]   # True Negative Rate (TNR)
fpr_test = 1 - spec_test        # False Positive Rate (FPR)

# AUC (Area Under Curve) 
sort_idx = np.argsort(fpr_test)
auc_test = np.trapezoid(sens_test[sort_idx], fpr_test[sort_idx])

# plot
plt.figure(figsize=(8, 6))

# Plot ROC of Test Set
plt.plot(fpr_test, sens_test, color='darkorange', lw=2, label=f'ROC curve Test (AUC = {auc_test:.3f})')

# Plot of the Random Classifier
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.500)')

#optimal threshold
opt_fpr = 1 - spec_test[idx_optimal_threshold]
opt_tpr = sens_test[idx_optimal_threshold]

plt.plot(opt_fpr, opt_tpr, marker='o', color='red', markersize=8, 
         label=f'Soglia Ottimale Youden ({optimal_threshold:.2f})')

# axises configuration
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) - FFR Detection')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# show the plot
plt.show()
