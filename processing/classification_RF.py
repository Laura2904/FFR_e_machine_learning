import numpy as np

#Random Forest implementation

def Gini_index(data, binary_class,splitting_criterion):
    """ It calculates the impurity variation between the
      parent node and child node"""

    right_node = binary_class[data >= splitting_criterion]
    left_node = binary_class[data < splitting_criterion]

    parentclass = np.mean(binary_class)
    parent_impurity = (1-parentclass)*parentclass

    if right_node.size == 0:
        impurity_right_node = 0
        p_right_node = 0
    else:
        binary_class_right = np.mean(right_node)
        impurity_right_node = (1- binary_class_right) * binary_class_right
        p_right_node = np.size(right_node) / np.size(binary_class)

    if left_node.size == 0:
        impurity_left_node = 0
        p_left_node = 0
    else:
        binary_class_left = np.mean(left_node)
        impurity_left_node = (1- binary_class_left) * binary_class_left
        p_left_node = np.size(left_node) / np.size(binary_class)

    delta_impurity = parent_impurity - (p_left_node*impurity_left_node + 
                                      p_right_node*impurity_right_node)
    
    return delta_impurity


def best_spltting_criterion(data,binary_class):
    """ It finds the best splitting criterion"""

    number_of_features = data.shape[1]
    best_gain = -np.inf
    best_feature = 0
    best_split = 0

    for i in range(number_of_features):
        val = np.unique(data[:, i])

        for j in range(len(val)-1):
            splitting = (val[j] + val[j+1])/2
            splitting_value = data[:,i]

            gain = Gini_index(splitting_value, binary_class,splitting)

            if gain > best_gain:
                best_gain = gain
                best_feature = i
                best_split = splitting

    best_splitting = [best_feature, best_split]

    return best_splitting

