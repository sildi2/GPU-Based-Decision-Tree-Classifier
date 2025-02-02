import pyopencl as cl
import numpy as np
import torch
from DecisionTree.Node import Node

class DecisionTreeClassifierGPU:
    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

    def build_tree(self, dataset, curr_depth=0):
        ''' Recursive function to build the tree using OpenCL '''
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = X.shape

        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(X, Y, num_samples, num_features)

            if best_split is None or "info_gain" not in best_split or best_split["info_gain"] <= 0:
                return Node(value=self.calculate_leaf_value(Y))

            left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
            right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)

            return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree,
                        best_split["info_gain"])

        return Node(value=self.calculate_leaf_value(Y))

    def get_best_split(self, X, Y, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")

        X = np.ascontiguousarray(X, dtype=np.float32)
        Y = np.ascontiguousarray(Y, dtype=np.float32)

        mf = cl.mem_flags
        X_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
        Y_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Y)


    def calculate_leaf_value(self, Y):
        ''' Compute the most frequent class '''
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, Y):
        ''' Train the decision tree '''
        dataset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        ''' Predict new samples '''
        predictions = [self.make_prediction(x, self.root) for x in X]
        return np.array(predictions)

    def make_prediction(self, x, tree):
        ''' Predict a single sample '''
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
