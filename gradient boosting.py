import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

# Optional: Plot the results
import matplotlib.pyplot as plt

import seaborn as sns



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
            
            tree=DecisionTreeRegressor(max_depth=max_depth)
            
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
X = np.random.rand(1000, 1)  
y = np.cos(2 * np.pi * X).ravel() + np.random.randn(1000) * 0.1  # noisy sine wave

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor()
n_estimators=100
gbr.fit(X_train, y_train,n_estimators=n_estimators,max_depth=3, lr=0.1)

predictions,L,L_test = gbr.predict(X_test,y_test)

plt.scatter(X_train, y_train, label="Training data", color="blue")
plt.scatter(X_test, y_test, label="Test data", color="green")
plt.scatter(X_test, predictions, label="Predictions", color="red")
plt.legend()
plt.show()     


sns.lineplot(x=np.linspace(0,n_estimators+1,n_estimators),y=L, label='Train')

sns.lineplot(x=np.linspace(0,n_estimators+1,n_estimators),y=L_test, label='Test')

    
    