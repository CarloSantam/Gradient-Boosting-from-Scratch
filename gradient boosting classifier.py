import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_classif
# Optional: Plot the results
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import pandas as pd





class GradientBoostingClassifier:
    
   def sigmoid_function(self, x): 
        if x >= 0:
            z = np.exp(-x)
            
            return 1/(1+z)
        
        else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
            z = np.exp(x)
            
            return z / (1 + z)
    
     
   def sigmoid(self, x):
        return np.array([self.sigmoid_function(value) for value in x])
        
        
   def loss(self,pred,y):
        N=len(y)   
        predict_1 = y * np.log(pred)
        predict_0 = (1 - y) * np.log(1-pred)    
        return -np.sum(predict_1 + predict_0)/N
    
   def pseudoresiduals(self,pred,y):
        res=(y-pred)
        return res
    
   def fit(self,X,y,n_estimators, max_depth, lr):
          
        logodds=np.log(len(y[y==1])/len(y[y==0]))

        logoddsprediction=logodds*np.ones(len(y))
        
        logodds_start=logodds
         
        prob=self.sigmoid_function(logodds)*np.ones(len(y))
                
        initial_prediction=prob
                                
        trees=[]
        
        Loss=np.random.rand(n_estimators)
              
        res=self.pseudoresiduals(prob,y)
                
        
        for k in range(0,n_estimators):
                        
            tree=DecisionTreeRegressor(max_depth=max_depth)
            
            tree.fit(X,res)
            
            ids = tree.apply(X)  
            
            # looping through the terminal nodes 
            for j in np.unique(ids):
                
                            
              fltr = ids==j

              num = np.sum(res[fltr])
              den = np.sum(prob[fltr]*(1-prob[fltr]))
                            
              gamma = num / den
                 
              logoddsprediction[fltr] += lr * gamma
              
              tree.tree_.value[j] = gamma

            prob=self.sigmoid(logoddsprediction)
                        
            res=self.pseudoresiduals(prob,y)

            L=self.loss(prob,y)
                                    
            trees.append(tree)
            
            Loss[k]=L
            
        self.trees=trees
        
        self.Loss=Loss
        
        self.lr=lr
        
        self.initial_prediction=initial_prediction
        
        self.gamma=gamma
        
        self.logodds_start=logodds_start

        
   def predict(self, X,y=None):
                
        L_test=[]
        
        logoddspred=self.logodds_start
        
        for tree in self.trees:
            
            logoddspred+= self.lr * tree.predict(X)

            
            predictions=self.sigmoid(logoddspred)

            try:
                L_test1=self.loss(predictions,y)
                
                L_test.append(L_test1)
                
            except:
                pass
            
        return [1 if i > 0.5 else 0 for i in predictions],self.Loss, L_test
    

########################################## pokemon example

Data=pd.read_csv("pokemon.csv")

X = Data.drop(columns=['is_legendary']).select_dtypes(include=['number'])
y = Data['is_legendary']
feature_names = X.columns

X=X.fillna(method='ffill')

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=45)

mi=mutual_info_classif(X_train, y_train, discrete_features='auto',random_state=45)

sorted_indices = np.argsort(mi)[::-1]

Feature_selected=6

sorted_indices=sorted_indices[:Feature_selected]

X_train=X_train[:,sorted_indices]

X_test=X_test[:,sorted_indices]

sorted_pairs = sorted(zip(mi, feature_names),reverse=True)
x_sorted, y_sorted = zip(*sorted_pairs)

fig, axs = plt.subplots(1,figsize=(10, 5))

sns.barplot(x=np.array(x_sorted)[1:Feature_selected], y=np.array(y_sorted)[1:Feature_selected])

plt.xlabel('Mutual Information')

plt.ylabel('Features')

plt.show()

gbm = GradientBoostingClassifier()

n_estimators=1000
gbm.fit(X_train, y_train,n_estimators=n_estimators,max_depth=1, lr=0.01)

predictions,L,L_test = gbm.predict(X_test,y_test)

sns.lineplot(x=np.linspace(0,n_estimators+1,n_estimators),y=L, label='Train')

sns.lineplot(x=np.linspace(0,n_estimators+1,n_estimators),y=L_test, label='Test')

conf_matrix = confusion_matrix(y_test, predictions)

plot_confusion_matrix(conf_matrix, show_absolute=True, show_normed=False)

plt.title('Confusion Matrix')





