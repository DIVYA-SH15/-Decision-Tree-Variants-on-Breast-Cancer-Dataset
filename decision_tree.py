import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# -----------------------------
# Load Dataset
# -----------------------------
data = load_breast_cancer()

X = pd.DataFrame(data.data,columns=data.feature_names)
y = pd.Series(data.target)

print(X.head())
print("\nDataset Shape:",X.shape)


# -----------------------------
# Train Test Split
# -----------------------------
X_train,X_test,y_train,y_test = train_test_split(
X,y,
test_size=0.2,
random_state=42
)


# -----------------------------
# Model Implementations
# -----------------------------

models={

"CART_GINI":
DecisionTreeClassifier(criterion='gini'),

"ID3_ENTROPY":
DecisionTreeClassifier(criterion='entropy'),

"MAX_DEPTH":
DecisionTreeClassifier(max_depth=5),

"MIN_SPLIT":
DecisionTreeClassifier(min_samples_split=10),

"MIN_LEAF":
DecisionTreeClassifier(min_samples_leaf=5),

"PRUNED_TREE":
DecisionTreeClassifier(ccp_alpha=0.01),

"EXTRA_TREES":
ExtraTreesClassifier(n_estimators=100)
}


results=[]

for name,model in models.items():

    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    acc=accuracy_score(y_test,pred)

    results.append([name,acc])

    print(name,"Accuracy =",acc)



# -----------------------------
# Result Table
# -----------------------------
result_df=pd.DataFrame(
results,
columns=["Model","Accuracy"]
)

print("\n")
print(result_df)



# -----------------------------
# Confusion Matrix for CART
# -----------------------------
cart=DecisionTreeClassifier()
cart.fit(X_train,y_train)

pred=cart.predict(X_test)

cm=confusion_matrix(y_test,pred)

print("\nConfusion Matrix")
print(cm)



# -----------------------------
# Accuracy Graph
# -----------------------------
plt.figure(figsize=(10,5))

plt.bar(result_df["Model"],
        result_df["Accuracy"])

plt.title("Comparison of 7 Decision Tree Variants")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.show()