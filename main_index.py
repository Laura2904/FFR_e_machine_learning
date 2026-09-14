# Main for Auditory Capacity index assessment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import processing.Auditory_Capacity.assessment_index
from processing.Auditory_Capacity.indexes import Auditory_Capacity_Index as aci

# ============================================================
# DATA LOADING
# ============================================================

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
features_names = df.columns[2:-1].to_list()

# ============================================================
# BARTLETT'S TEST FOR SPHERICITY
# ============================================================

chi_square, p_value, degrees_of_freedom = processing.Auditory_Capacity.assessment_index.bartlett_test_sphericity(data)

print(
    f"Bartlett's test for sphericity:\n"
    f"Chi-square statistic: {chi_square}\n"
    f"Degrees of freedom: {degrees_of_freedom}\n"
    f"P-value: {p_value}"
)

# ============================================================
# AUDITORY CAPACITY INDEXES
# ============================================================

ACI1, ACI2, GACI = aci(data,required_variance=0.90)

# ============================================================
# RESULTS TABLE
# ============================================================

sogg = df.iloc[:, 0].to_numpy()

result = pd.DataFrame({
    "subject": sogg,
    "index_periodic_encoding": ACI1,
    "transitory_encoding": ACI2,
    "Global index": GACI,
    "Classe": binary_class
})

print("\nResults:")
print(result)

# ============================================================
# MEAN INDEX FOR EACH CLASS
# ============================================================

classes = np.unique(binary_class)

mean_GACI = np.zeros(len(classes))
std_GACI = np.zeros(len(classes))

mean_ACI1 = np.zeros(len(classes))
std_ACI1 = np.zeros(len(classes))

mean_ACI2 = np.zeros(len(classes))
std_ACI2 = np.zeros(len(classes))


for i, class_i in enumerate(classes):

    idx = binary_class == class_i

    mean_GACI[i] = np.mean(GACI[idx])
    std_GACI[i] = np.std(GACI[idx], ddof=1)

    mean_ACI1[i] = np.mean(ACI1[idx])
    std_ACI1[i] = np.std(ACI1[idx], ddof=1)

    mean_ACI2[i] = np.mean(ACI2[idx])
    std_ACI2[i] = np.std(ACI2[idx], ddof=1)


table_average_classes = pd.DataFrame({
    "Classe": classes,
    "Media_Global_Index": mean_GACI,
    "Std_Global_Index": std_GACI,
    "Media_Index1": mean_ACI1,
    "Std_Index1": std_ACI1,
    "Media_Index2": mean_ACI2,
    "Std_Index2": std_ACI2
})


print("\nMean indexes by class:")
print(table_average_classes)


# ============================================================
# BOXPLOT
# ============================================================

boxplot_data = []
boxplot_labels = []

for class_i in classes:

    idx = binary_class == class_i

    boxplot_data.append(GACI[idx])
    boxplot_data.append(ACI1[idx])
    boxplot_data.append(ACI2[idx])

    boxplot_labels.append(f"{class_i}-GI")
    boxplot_labels.append(f"{class_i}-ACI1")
    boxplot_labels.append(f"{class_i}-ACI2")


plt.figure(figsize=(10, 6))

plt.boxplot(boxplot_data,tick_labels=boxplot_labels)

plt.xlabel("CLasses and indexes")
plt.ylabel("Index values")
plt.title("Boxplot of Auditory Capacity Indexes by Class")

plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# ============================================================
# correlation test between indexes and features
# ============================================================

corr_ACI1, corr_ACI2, corr_GACI = processing.Auditory_Capacity.assessment_index.correlation_test(ACI1, ACI2, GACI, data)

# ============================================================
# CORRELATION TABLE
# ============================================================

table_corr = pd.DataFrame({
    "Feature_FFR": features_names,
    "ACI1": corr_ACI1,
    "ACI2": corr_ACI2,
    "GACI": corr_GACI
})

print("\nCorrelations:")
print(table_corr)


# ============================================================
# CORRELATION BAR PLOT
# ============================================================

x = np.arange(len(features_names))
width = 0.25

plt.figure(figsize=(12, 6))

plt.bar(x - width,corr_ACI1,width,label="ACI1")

plt.bar(x,corr_ACI2,width,label="ACI2")

plt.bar(x + width,corr_GACI,width,label="Global index")

plt.xlabel("Feature of FFR")
plt.ylabel("Correlation coefficient")
plt.title("Correlation between ICU indices and FFR features")

plt.xticks(x,features_names,rotation=45,ha="right")

plt.legend()
plt.grid(axis="y")

plt.tight_layout()
plt.show()

# ============================================================
# RELIABILITY TEST 
# ===========================================================
data1 = np.column_stack((data, binary_class))
pc_mean, pc_std, corr_ACI1, corr_ACI2, corr_GACI,ACI1_test,ACI2_test,GACI_test = processing.Auditory_Capacity.assessment_index.reliability_test(data1)

print(f"\nReliability Test Results:")
print(f"Mean number of principal components: {pc_mean:.2f} (Std: {pc_std:.2f})")
print(f"Correlation ACI1: {corr_ACI1:.3f}")
print(f"Correlation ACI2: {corr_ACI2:.3f}")
print(f"Correlation GACI: {corr_GACI:.3f}")

processing.Auditory_Capacity.assessment_index.plot_reliability(
    ACI1,
    ACI1_test,
    corr_ACI1,
    "Tracking encoding",
    "Reliability ACI1"
)

processing.Auditory_Capacity.assessment_index.plot_reliability(
    ACI2,
    ACI2_test,
    corr_ACI2,
    "Transitory encoding",
    "Reliability ACI2"
)

processing.Auditory_Capacity.assessment_index.plot_reliability(
    GACI,
    GACI_test,
    corr_GACI,
    "Global index",
    "Reliability Global index"
)