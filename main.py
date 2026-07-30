import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from processing.classification.logistic_regression import binomial_logistic_regression as BLR
from processing.classification import random_forest as RF

#%% Data loading
df = pd.read_excel(
    "data/CA_normo.xlsx",
    sheet_name="sw1000",
    header=0
)

#%% feature and class separation
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

#set up
k = 5
alpha = np.linspace(1, 0, 101)

#LOGISTIC REGRESSION
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
idx_opt_LR = np.argmax(youden_index)
optimal_threshold_LR= alpha[idx_opt_LR]
max_youden_val = youden_index[idx_opt_LR]

print(f"The best threshold(Alpha): {optimal_threshold_LR:.3f}")
print(f"Best Youden value: {max_youden_val:.3f}")
print(f"Accuracy Test (the best threshold):  {Accuracy_LR[1][idx_opt_LR]:.3f}")
print(f"Sensibility Test (the best threshold): {Sensibility_LR[1][idx_opt_LR]:.3f}")
print(f"Specificity Test (the best threshold): {Specificity_LR[1][idx_opt_LR]:.3f}")

#%% RANDOM FOREST
# array of trees to find the optimal number
trees = [10, 30, 50]

n_trees_opt, Accuracy_RF, Sensibility_RF, Specificity_RF = RF.run_random_forest_pipeline(
    training_set, test_set, k, trees
)

Acc_RF_train, Acc_RF_test = Accuracy_RF[:, 0], Accuracy_RF[:, 1]
Sens_RF_train, Sens_RF_test = Sensibility_RF[:, 0], Sensibility_RF[:, 1]
Spec_RF_train, Spec_RF_test = Specificity_RF[:, 0], Specificity_RF[:, 1]

print("\n--- Random Forest results ---")
print(f"Optimal number of trees selected: {n_trees_opt}")
print(f"Accuracy Train (threshold 0.5): {Acc_RF_train[50]:.3f}")
print(f"Accuracy Test (threshold 0.5):  {Acc_RF_test[50]:.3f}")
print(f"Sensibility Test (threshold 0.5): {Sens_RF_test[50]:.3f}")
print(f"Specificity Test (threshold 0.5): {Spec_RF_test[50]:.3f}")

# Find best threshold RF (Youden Index)
youden_RF = (Sens_RF_train + Spec_RF_train) - 1
idx_opt_RF = np.argmax(youden_RF)
opt_thresh_RF = alpha[idx_opt_RF]

print(f"\nBest threshold RF (Alpha): {opt_thresh_RF:.3f}")
print(f"Accuracy Test RF (best threshold):  {Acc_RF_test[idx_opt_RF]:.3f}")
print(f"Sensibility Test RF (best threshold): {Sens_RF_test[idx_opt_RF]:.3f}")
print(f"Specificity Test RF (best threshold): {Spec_RF_test[idx_opt_RF]:.3f}")



# ROC

#metrics LR
sens_test_LR = Sensibility_LR[1]   # True Positive Rate (TPR)
spec_test_LR = Specificity_LR[1]   # True Negative Rate (TNR)
fpr_test_LR = 1 - spec_test_LR        # False Positive Rate (FPR)
sort_idx_LR = np.argsort(fpr_test_LR)
auc_test_LR = np.trapezoid(sens_test_LR[sort_idx_LR], fpr_test_LR[sort_idx_LR])

# Metrics RF
fpr_test_RF = 1 - Spec_RF_test
sort_idx_RF = np.argsort(fpr_test_RF)
auc_test_RF = np.trapezoid(Sens_RF_test[sort_idx_RF], fpr_test_RF[sort_idx_RF])


# Figure Plotting
plt.figure(figsize=(9, 7))

# ROC Logistic Regression
plt.plot(fpr_test_LR, sens_test_LR, color='darkorange', lw=2, 
         label=f'ROC Logistic Regression (AUC = {auc_test_LR:.3f})')

# ROC Random Forest
plt.plot(fpr_test_RF, Sens_RF_test, color='forestgreen', lw=2, 
         label=f'ROC Random Forest ({n_trees_opt} trees) (AUC = {auc_test_RF:.3f})')

# Random Classifier Baseline
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier (AUC = 0.500)')

# Optimal Threshold Points
opt_fpr_LR = 1 - spec_test_LR[idx_opt_LR]
opt_tpr_LR = sens_test_LR[idx_opt_LR]
plt.plot(opt_fpr_LR, opt_tpr_LR, marker='o', color='red', markersize=8, 
         label=f'Youden Opt. LR ({optimal_threshold_LR:.2f})')

opt_fpr_RF = 1 - Spec_RF_test[idx_opt_RF]
opt_tpr_RF = Sens_RF_test[idx_opt_RF]
plt.plot(opt_fpr_RF, opt_tpr_RF, marker='s', color='darkred', markersize=8, 
         label=f'Youden Opt. RF ({opt_thresh_RF:.2f})')

# Axis configuration
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve Comparison: Logistic Regression vs Random Forest')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Show the plot
plt.show()