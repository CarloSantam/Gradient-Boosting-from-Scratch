import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error

# Optional: Plot the results
import matplotlib.pyplot as plt

import seaborn as sns

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class RegressorTree:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples >= self.min_samples_split and depth < self.max_depth:
            best_split = self._best_split(X, y, n_features)
            if best_split:
                left_idx, right_idx = best_split['indices']
                left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
                right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
                return Node(best_split['feature'], best_split['threshold'], left, right)
        return Node(value=np.mean(y))

    def _best_split(self, X, y, n_features):
        best_mse = float('inf')
        split = {}
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = np.where(X[:, feature] <= threshold)[0]
                right_idx = np.where(X[:, feature] > threshold)[0]
                if len(left_idx) > 0 and len(right_idx) > 0:
                    mse = self._calculate_mse(y[left_idx], y[right_idx])
                    if mse < best_mse:
                        best_mse = mse
                        split = {
                            'feature': feature,
                            'threshold': threshold,
                            'indices': (left_idx, right_idx)
                        }
        #return split if split else None
        
        return split

    def _calculate_mse(self, left_y, right_y):
        
        left_mse = np.var(left_y) * len(left_y)
        right_mse = np.var(right_y) * len(right_y)
        return (left_mse + right_mse) / (len(left_y) + len(right_y))
    
    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.root
        while node.value is None:
            if inputs[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class GradientBoostingRegressor:
    
    def loss(self,pred,y):
        L=1/2*np.sum((pred-y)**2)
        return L
    
    def pseudoresiduals(self,pred,y):
        res=(y-pred)
        return res
    
    def fit(self,X,y,n_estimators, max_depth, lr):
        initial_prediction = np.mean(y)
        predictions = np.full(y.shape, initial_prediction)
                
        trees=[]
        
        Loss=np.random.rand(n_estimators)
        
        
        for k in range(0,n_estimators):
            
            tree=RegressorTree(max_depth=max_depth)
            
            res=self.pseudoresiduals(predictions,y)

            tree.fit(X, res)
            
            predictions=predictions+lr*tree.predict(X)
            
            L=self.loss(predictions,y)
                        
            trees.append(tree)
            
            Loss[k]=L
            
        self.trees=trees
        self.Loss=Loss
        
        self.lr=lr
        
        self.initial_prediction=initial_prediction
        
    def predict(self, X,y=None):
        
        predictions=np.full(X.shape[0], self.initial_prediction)
        L_test=[]
        

        
        for tree in self.trees:
            
            predictions=predictions+self.lr*tree.predict(X)
            
            try:
                L_test1=self.loss(predictions,y)
                
                L_test.append(L_test1)
                
            except:
                pass
            
        return predictions,self.Loss, L_test


np.random.seed(42)
X = np.random.rand(1000, 1)  # 100 samples, 1 feature
y = np.cos(2 * np.pi * X).ravel() + np.random.randn(1000) * 0.1  # noisy sine wave

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the GradientBoostingRegressor
gbr = GradientBoostingRegressor()
n_estimators=100
gbr.fit(X_train, y_train,n_estimators=n_estimators,max_depth=3, lr=0.1)

# Make predictions and evaluate the model
predictions,L,L_test = gbr.predict(X_test,y_test)




plt.scatter(X_train, y_train, label="Training data", color="blue")
plt.scatter(X_test, y_test, label="Test data", color="green")
plt.scatter(X_test, predictions, label="Predictions", color="red")
plt.legend()
plt.show()     


sns.lineplot(x=np.linspace(0,n_estimators+1,n_estimators),y=L, label='Train')

sns.lineplot(x=np.linspace(0,n_estimators+1,n_estimators),y=L_test, label='Test')

    
    