from sklearn.model_selection import KFold
import numpy as np
from collections import deque


class XGBoostNode:
    def __init__(self, similarity=float('-inf'), parent=None, left=None, right=None,
                 feature_index=None, indices=None, output_val=None, split_value=None) -> None:
        self.similarity = similarity
        self.parent = parent
        self.left = left
        self.right = right
        self.split_value = split_value
        self.feature_index = feature_index
        self.indices = [] if indices is None else indices
        self.output_val = output_val

    def __str__(self):
        return f"Split Value:{self.split_value}"


class XGBoostRegression:

    # Predict the residuals of the previous tree

    def __init__(self, learning_rate: float = 0.1, gamma: float = 50, lmbda: float = 1.0, max_depth: int = 4, max_leaves: int = 4, n_trees: int = 4) -> None:
        self.lmbda = lmbda
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.gamma = gamma
        self.trees = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.initial_guess = np.mean(y)

    def predict(self, X):
        # Aggregate predictions from all trees
        predictions = np.full(len(X), self.initial_guess)
        for tree in self.trees:
            for i, data_point in enumerate(X):
                predictions[i] += self._traverse_tree(tree, data_point)
        return predictions

    def train(self, X, y):
        # Calculate initial residuals
        residuals = y - self.initial_guess

        # Loop to create n trees
        next_guesses = residuals
        for tree_i in range(self.n_trees):
            # Create the tree
            new_tree = self._build_tree(residuals)

            # Save the tree
            self.trees.append(new_tree)

            # Update predictions and residuals
            for i in range(len(X)):
                prediction = self.learning_rate*self._traverse_tree(
                    new_tree, X[i]) + next_guesses[i]
                residuals[i] = y[i] - prediction

            # Update the initial guess for the next tree
            next_guesses = residuals

    def _prune_tree(self, node, residuals):
        # Base case: if the node is a leaf
        if node.left is None and node.right is None:
            return self._calc_similarity_score(residuals[node.indices])

        # Recursively prune the left and right subtrees
        left_gain = self._prune_tree(node.left, residuals) if node.left else 0
        right_gain = self._prune_tree(
            node.right, residuals) if node.right else 0

        # Calculate total gain with this split
        total_gain = left_gain + right_gain - self.gamma

        # Prune if gain is not sufficient
        if total_gain < 0:
            node.left = None
            node.right = None
            node.output_val = np.sum(residuals[node.indices]) / \
                (len(node.indices) + self.lmbda)
            return 0

        return total_gain

    def _build_tree(self, residuals):
        # Randomly select feature to sort by initially
        rand_i = np.random.randint(0, self.X.shape[1])

        # Sort indices based on the selected feature
        sorted_indices = np.argsort(self.X[:, rand_i])

        # Create root node with sorted indices
        new_tree = XGBoostNode(similarity=self._calc_similarity_score(
            residuals), indices=sorted_indices.tolist())

        # Create queues to store nodes to check
        current_level_queue = deque()
        next_level_queue = deque()

        current_level_queue.append(new_tree)

        # Iterate until max depth
        current_depth = 1
        while current_depth < self.max_depth:
            while current_level_queue:
                curr_node = current_level_queue.pop()

                # Randomly select feature to split by
                rand_i = np.random.randint(0, self.X.shape[1])
                curr_node.feature_index = rand_i

                # Split on feature
                left, right = self._split(residuals, curr_node, rand_i)

                # Append children to next level queue if they exist
                if left:
                    next_level_queue.append(left)

                if right:
                    next_level_queue.append(right)

            # Move to the next level
            current_level_queue, next_level_queue = next_level_queue, deque()
            current_depth += 1

        # Process remaining nodes in the next level queue as leaves
        while current_level_queue:
            leaf_node = current_level_queue.pop()
            leaf_node.output_val = np.sum(residuals[leaf_node.indices]) / \
                (len(leaf_node.indices) + self.lmbda)

        # Post-prune the tree after building
        self._prune_tree(new_tree, residuals)

        return new_tree

    def _traverse_tree(self, tree, data):
        curr_node = tree
        while curr_node.output_val == None:
            if data[curr_node.feature_index] <= curr_node.split_value:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
        return curr_node.output_val

    def _split(self, residuals, node, feat_i):
        # check if we can even split
        if len(node.indices) <= 1:
            # set output value
            node.output_val = np.sum(residuals[node.indices]) / \
                (len(node.indices) + self.lmbda)
            return None, None

        # Grab the values for the given feature
        vals = self.X[node.indices, feat_i]
        local_residuals = residuals[node.indices]

        # Get similarity score for root node
        root_sim = node.similarity

        best_gain = float('-inf')
        left = None
        right = None
        # Try the split points for each value
        for i in range(len(vals) - 1):
            # Get split residual values
            left_res = local_residuals[:i + 1]
            right_res = local_residuals[i + 1:]

            # Calculate similarity for left and right nodes
            left_sim = self._calc_similarity_score(left_res)
            right_sim = self._calc_similarity_score(right_res)

            # Calculate gain for given split
            gain = self._calc_gain(root_sim, left_sim, right_sim)

            # Compare to best gain
            if gain > best_gain:
                best_gain = gain

                # Extract the indices for left and right nodes from the node's indices
                left_indices = node.indices[:i + 1]
                right_indices = node.indices[i + 1:]

                # Create the left and right nodes
                left = XGBoostNode(similarity=left_sim,
                                   parent=node, indices=left_indices)
                right = XGBoostNode(similarity=right_sim,
                                    parent=node, indices=right_indices)

                # set split value for current node
                node.split_value = vals[i]

        # Update children for root node
        node.left = left
        node.right = right

        return left, right

    def _calc_gain(self, root_sim, left_sim, right_sim):
        return left_sim + right_sim - root_sim

    def _calc_similarity_score(self, residuals):
        # get sum of residuals squared
        res_sq = np.square(np.sum(residuals))

        # get number of residuals + lambda/regularization
        n_residuals = len(residuals) + self.lmbda

        return res_sq / n_residuals


# Define a simple function to print the tree
def print_tree(node, depth=0):
    if node is not None:
        # Print the current node's details
        print(" " * 4 * depth +
              f"Depth {depth}: Node(similarity={node.similarity}, feature_index={node.feature_index}, output_val={node.output_val}, indices={node.indices})")

        # Recursively print the left and right children
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)


# Sample Data
X = np.array([[-5], [10], [8], [-3]])
y = np.array([-10, 6, 7, -7])

# Testing the model
xgb_model = XGBoostRegression(max_depth=3, n_trees=4, gamma=0, lmbda=0)
xgb_model.fit(X, y)
xgb_model.train(X, y)

# Print the structure of each built tree
for i, tree in enumerate(xgb_model.trees):
    print(f"\nTree {i}:")
    print_tree(tree)

xgb_model._traverse_tree(xgb_model.trees[0], [-3])

# Test predictions
predictions = xgb_model.predict(X)
for data_point, pred in zip(X, predictions):
    print(f"Prediction for {data_point}: {pred}")


# Function to generate synthetic data
def generate_data(n_samples=100):
    # Generate random X values
    X = np.random.uniform(-10, 10, size=(n_samples, 1))

    # Generate y values based on a simple pattern and add some noise
    y = X[:, 0] * 2 + np.random.normal(0, 2, n_samples)  # y = 2x + noise

    return X, y


# Generate data
X, y = generate_data(n_samples=1000)

xgb_model = XGBoostRegression(max_depth=6, n_trees=10, gamma=20, lmbda=0)
xgb_model.fit(X, y)
xgb_model.train(X, y)


# Print the structure of each built tree (optional, can be commented out if too verbose)
# for i, tree in enumerate(xgb_model.trees):
#     print(f"\nTree {i}:")
#     print_tree(tree)

# Test predictions
predictions = xgb_model.predict(X)
for i in range(10):  # Print predictions for the first 10 data points
    print(f"Prediction for {X[i]}: {predictions[i]}, Actual: {y[i]}")


def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error between the true and predicted values.

    :param y_true: Array of true target values.
    :param y_pred: Array of predicted target values.
    :return: Mean squared error.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


# Function to perform cross-validation

def cross_val_score(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        model.train(X_train, y_train)

        # Implement a scoring function, e.g., mean squared error
        predictions = model.predict(X_test)
        # Define this function as per your requirement
        score = mean_squared_error(y_test, predictions)
        scores.append(score)

    return np.mean(scores)


# # Grid search
# gamma_values = [0.1, 0.3, 0.5, 0.7, 1.0]
# lambda_values = [0.1, 0.3, 0.5, 0.7, 1.0]

# best_score = float('inf')
# best_params = {}

# for gamma in gamma_values:
#     for lambda_val in lambda_values:
#         model = XGBoostRegression(
#             gamma=gamma, lmbda=lambda_val, max_depth=3, n_trees=10)

#         score = cross_val_score(model, X, y, n_splits=5)

#         if score < best_score:
#             best_score = score
#             best_params = {'gamma': gamma, 'lambda': lambda_val}

# print("Best Score:", best_score)
# print("Best Parameters:", best_params)
