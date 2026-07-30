import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from typing import List, Tuple, Optional
from assessment import assessment_metrics
from classification.decision_tree import DecisionTree


class RandomForestClassifier:
    """Random Forest model"""

    def __init__(self, n_estimators: int = 100, max_depth: int = 20,
                  min_samples_leaf: int = 5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees: List[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: Optional[int] = None):
        """TBoostrap for training"""
        if seed is not None:
            np.random.seed(seed)

        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            # Bootstrap 
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[boot_idx], y[boot_idx]

            #Decision Tree initialization
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features='sqrt'
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """compute the mean probability through the Trees aggregation"""
        #collectt the prediction from each tree
        tree_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        # compute the mean
        return np.mean(tree_probs, axis=0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Class is precited based on a specific threshold"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


def run_random_forest_pipeline(trainingdata: np.ndarray, testdata: np.ndarray, 
    k: int, trees: List[int] ):

    """ Building of Random Forest"""
    
    # class extraction
    X_train = trainingdata[:, :-1]
    y_train = trainingdata[:, -1].astype(int)

    X_test = testdata[:, :-1]
    y_test = testdata[:, -1].astype(int)

    # 2. CROSS-VALIDATION to find the optimal number of trees
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    acc_cv_for_n_trees = []

    for n_trees in trees:
        acc_fold = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr_fold, y_tr_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

            # Initialization and training of RF
            rf = RandomForestClassifier(n_estimators=n_trees)
            rf.fit(X_tr_fold, y_tr_fold, seed=42)

            # Prediction 
            y_pred_val = rf.predict(X_val_fold, threshold=0.5)

            # Accuracy
            acc = np.mean(y_pred_val == y_val_fold)
            acc_fold.append(acc)

        # mean accuracy 
        acc_cv_for_n_trees.append(np.mean(acc_fold))

    # selection of the best number of trees
    best_idx = int(np.argmax(acc_cv_for_n_trees))
    n_optimal_trees = trees[best_idx]

    # training in the whole training data set
    final_rf = RandomForestClassifier(n_estimators=n_optimal_trees)
    final_rf.fit(X_train, y_train, seed=42)

    # Probability for training set and test set
    score_train = final_rf.predict_proba(X_train)
    score_test = final_rf.predict_proba(X_test)

    # Metrics based on different thresholds
    thresholds = np.linspace(1.0, 0.0, 101)

    acc_train, sens_train, spec_train = assessment_metrics(y_train, score_train, thresholds)
    acc_test, sens_test, spec_test = assessment_metrics(y_test, score_test, thresholds)

    # Matrixes for metrics
    Accuracy= np.column_stack((acc_train, acc_test))
    Sensibility = np.column_stack((sens_train, sens_test))
    Specificity = np.column_stack((spec_train, spec_test))

    return n_optimal_trees, Accuracy, Sensibility, Specificity