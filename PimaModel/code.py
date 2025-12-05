
"""<h1>Step 1: Import Libraries</h1>"""

import pandas as pd # create dataframes
import numpy as np # for computation
import seaborn as sns # data visualization also i.e bivariant analysis
import matplotlib.pyplot as plt
import sklearn as skl # all algos US, S
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

"""<h1>Step 2: Exclude Header files</h1>"""

column_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
data = pd.read_csv("pima-indians-diabetes.csv", header=None, names=column_names,skiprows=(0,0))

data.head()

data.info()

data.describe()

"""<h1>Step 3(optional but in some cases necessary): Conversion of string to numeric format(do if string input)</h1>"""

convert_col = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
for col in convert_col:
  data[col]= pd.to_numeric(data[col])

"""<h1>Step 4: Feature selection</h1>"""

feature_columns = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = data[feature_columns]
y = data.label

"""<h1>Step 5: Check Correlation</h1>"""

corr = data.corr()
plt.figure(figsize=(40,30))

coor_range = corr[(corr>=0.3)| (corr<=-0.1)]
sns.heatmap(coor_range, vmax=.8,linewidths=0.01, square=True, annot=True, cmap='GnBu', linecolor="white", cbar_kws={'label':'Feature correlation color'})
plt.title("correlation between features")
plt.ylabel("Y-axis")
plt.xlabel("x-axis")

"""<p>1 shows highly correlated</p>

<h1>Step 6: Train/Test split </h1>
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""<h1>Step 7: Apply Model</h1>"""

logistic_func = LogisticRegression()

"""<h1>Step 8: Fit Model with training data</h1>"""

logistic_func.fit(X_train, y_train)
y_prediction = logistic_func.predict(X_test)

"""<h1>Step 9: Model Evaluation</h1>"""

from sklearn import metrics
# using confusion matrix

cnf_confusion_matrix = metrics.confusion_matrix(y_test, y_prediction)
cnf_confusion_matrix

# Visualization of heatmap

class_name = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_name))
plt.xticks(tick_marks, class_name)
plt.yticks(tick_marks, class_name)


#creating heatmap for the confusion matrix

sns.heatmap(pd.DataFrame(cnf_confusion_matrix), annot=True,cmap="YlGnBu", fmt='g' ) #fmt=format
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix: Diabetes Patient', y=1.1)
plt.xlabel('predicted label')
plt.ylabel('actual label')

# confusion matrix conclusion for the evaluating metrics
print("Accuracy: ", metrics.accuracy_score(y_test, y_prediction))
print("Precision: ", metrics.precision_score(y_test, y_prediction))
print("Recall: ", metrics.recall_score(y_test, y_prediction))

print(y_test)
print(y_prediction)

"""<h1>ROC Curve</h1>"""

y_pred_probab = logistic_func.predict_proba(X_test)[::,1]
fpr, tpr,_ =metrics.roc_curve(y_test, y_pred_probab)
auc = metrics.roc_auc_score(y_test, y_pred_probab)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# F1 score
from sklearn.metrics import f1_score
f1_score(y_test, y_prediction, average=None)