#Here two classes are going to be created:
#class DecisionTreeNode which represents each node inside of the tree
#class DecisionTree which represents a collection of Decision Tree objects

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class DecisionTreeNode:
    leaf: bool = False
    probability: float = 0.0
    predicted_class: int = 0
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["DecisionTreeNode"] = None
    right: Optional["DecisionTreeNode"] = None


class DecisionTree:
    def __init__(self, max_depth: int = 20, min_samples_leaf: int = 5, max_features: Optional[str] = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # None or'sqrt' for Random Forest
        self.root: Optional[DecisionTreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(X, y, depth=0)

    # --- Splitting Methods ---

    def _gini_index(self, data_column: np.ndarray, binary_class: np.ndarray, threshold: float) -> float:
        """it computes the impurity reduction"""

        right_mask = data_column >= threshold
        left_mask = ~right_mask

        right_node = binary_class[right_mask]
        left_node = binary_class[left_mask]

        n_total = len(binary_class)
        if n_total == 0:
            return 0.0

        parent_p = np.mean(binary_class)
        parent_impurity = (1.0 - parent_p) * parent_p

        # Right node
        if right_node.size == 0:
            impurity_right, p_right = 0.0, 0.0
        else:
            p_right_class = np.mean(right_node)
            impurity_right = (1.0 - p_right_class) * p_right_class
            p_right = len(right_node) / n_total

        # Left Node
        if left_node.size == 0:
            impurity_left, p_left = 0.0, 0.0
        else:
            p_left_class = np.mean(left_node)
            impurity_left = (1.0 - p_left_class) * p_left_class
            p_left = len(left_node) / n_total

        return parent_impurity - (p_left * impurity_left + p_right * impurity_right)

    def _best_splitting_criterion(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best splitting feature"""

        n_features = X.shape[1]
        
        #In case of RF
        if self.max_features == 'sqrt':
            mtry = int(np.round(np.sqrt(n_features)))
            feature_indices = np.random.choice(n_features, size=mtry, replace=False)
        else:
            feature_indices = np.arange(n_features)

        best_gain = -np.inf
        best_feature = None
        best_split = None

        for i in feature_indices:
            vals = np.unique(X[:, i])
            if len(vals) <= 1:
                continue

            thresholds = (vals[:-1] + vals[1:]) / 2.0

            for split in thresholds:
                gain = self._gini_index(X[:, i], y, split)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_split = split

        return best_feature, best_split, best_gain

    # --- Build the tree---

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        depth += 1
        p_positive = np.mean(y)

        # pure leaf
        if p_positive == 1.0:
            return DecisionTreeNode(leaf=True, probability=1.0, predicted_class=1)
        elif p_positive == 0.0:
            return DecisionTreeNode(leaf=True, probability=0.0, predicted_class=0)

        # Stopping criterions
        if X.shape[0] <= self.min_samples_leaf or depth >= self.max_depth:
            return DecisionTreeNode(
                leaf=True,
                probability=float(p_positive),
                predicted_class=int(np.round(p_positive))
            )

        # Find the best splitting
        best_feature, best_split, best_gain = self._best_splitting_criterion(X, y)

        if best_feature is None or best_gain <= 0:
            return DecisionTreeNode(
                leaf=True,
                probability=float(p_positive),
                predicted_class=int(np.round(p_positive))
            )

        # Data splitting and recall method
        right_mask = X[:, best_feature] >= best_split
        left_mask = ~right_mask

        X_right, y_right = X[right_mask], y[right_mask]
        X_left, y_left = X[left_mask], y[left_mask]

        if len(y_right) == 0 or len(y_left) == 0:
            return DecisionTreeNode(
                leaf=True,
                probability=float(p_positive),
                predicted_class=int(np.round(p_positive))
            )

        return DecisionTreeNode(
            leaf=False,
            feature=best_feature,
            threshold=best_split,
            right=self._build_tree(X_right, y_right, depth),
            left=self._build_tree(X_left, y_left, depth)
        )

    # --- methods for prediction and probability---

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root)[0] for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root)[1] for x in X])

    def _predict_one(self, x: np.ndarray, node: DecisionTreeNode):
        if node.leaf:
            return node.probability, node.predicted_class
        
        if x[node.feature] >= node.threshold:
            return self._predict_one(x, node.right)
        else:
            return self._predict_one(x, node.left)