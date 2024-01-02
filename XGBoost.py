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


class XGBoostRegression:

    # Predict the residuals of the previous tree

    def __init__(self, gamma: float = 0.5, lmbda: float = 1.0, max_depth: int = 4, max_leaves: int = 4) -> None:
        self.lmbda = lmbda
        self.max_depth = max_depth

    def fit(self, X, y):
        self.X = X
        self.y = y

    def train(self, X, y):
        # calculate initial guess (mean of the target)
        intial_guess = np.mean(y)

        # calc initial residuals
        residuals = y - intial_guess

        # loop to create n trees
        for tree_i in range(self.n_trees):

            # create the tree
            self._build_tree()

            # get new predictions

            # compute the new residuals
            residuals = None

            # save the new residuals

            pass

        pass

    def _build_tree(self, residuals):
        # Randomly select feature to sort by initially
        rand_i = np.random.randint(0, self.X.shape[1])

        # Sort indices based on the selected feature
        sorted_indices = np.argsort(self.X[:, rand_i])

        # Create root node with sorted indices
        new_tree = XGBoostNode(indices=sorted_indices.tolist())

        # Create queue to store nodes to check
        queue = deque()
        queue.append(new_tree)

        # Iterate until max depth or can't split anymore
        for i in range(self.max_depth):
            if len(queue) == 0:
                break

            curr_node = queue.pop()

            # Randomly select feature to split by
            rand_i = np.random.randint(0, self.X.shape[1])

            # Split on feature
            left, right = self._split(residuals, curr_node, rand_i)

            # Append children to queue if they exist
            if left and left.output_val is None:
                queue.append(left)
            else:
                left.output_val = residuals[left.indices]/len(left.indices)

            if right and right.output_val is None:
                queue.append(right)
            else:
                left.output_val = residuals[right.indices]/len(right.indices)

        # we can reach max depth and not set our output values
        if left and left.output_val is None:
            queue.append(left)
        else:
            left.output_val = residuals[left.indices]/len(left.indices)

        if right and right.output_val is None:
            queue.append(right)
        else:
            left.output_val = residuals[right.indices]/len(right.indices)

        return new_tree

    def _calc_residuals(self, y, preds):
        pass

    def _split(self, residuals, node, feat_i):
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


# Sample Data
X = np.array([
    [15, 20, 35], [40, 55, 65], [70, 85, 90], [25, 30, 45]
])

# Sample residuals
residuals = np.array([-10.5, 6.5, 7.5, -7.5])

# Create an instance of the XGBoostTree with a max depth of, say, 3
xgb_tree = XGBoostRegression(X, max_depth=3)
xgb_tree.fit(X, 0)

# Build the tree
tree = xgb_tree._build_tree(residuals)
print(tree.left)
print(tree.right)