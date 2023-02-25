import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

myDf = pd.read_csv("KNN_Project_Data.csv")
#sns.pairplot(data=myDf, hue="TARGET CLASS", palette="coolwarm")
#plt.show()

#SCALE AND OPTIMIZE THE DATA
scaler = StandardScaler()
scaler.fit(myDf.drop("TARGET CLASS", axis=1))
scaled_features = scaler.transform(myDf.drop("TARGET CLASS", axis=1))
df_features = pd.DataFrame(data=scaled_features, columns=myDf.columns[:-1:])

#FEATURE VECTOR AND OUTPUTS TO TRAIN THE MODEL
X = df_features
y = myDf["TARGET CLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
error_rates = []

def optimal_k_finder(X_train, X_test, y_train, y_test, error_rates):
    """
    THE FUNCTION TO FIND OPTIMUM K VALUE FOR K-NEAREST-NEIGHBORS ALGORTIHM
    """
    for i in range(1,50):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X=X_train, y=y_train)
        pred_i = knn.predict(X=X_test)
        error_rates.append(np.mean(pred_i != y_test))

    op_k = error_rates.index(min(error_rates)) + 1 ## +1 SINCE INDEXING STARTS FROM 0 BUT K STARTS FROM 1 
    return op_k

optimal_k = optimal_k_finder(X_train, X_test, y_train, y_test, error_rates)
knn = KNeighborsClassifier(n_neighbors = optimal_k)
knn.fit(X=X_train, y=y_train)
predictions = knn.predict(X=X_test)

knn_confusionMatrix = confusion_matrix(y_true = y_test, y_pred=predictions)
knn_classRep = classification_report(y_true = y_test, y_pred=predictions)

print(knn_confusionMatrix)
print("")
print(knn_classRep)



