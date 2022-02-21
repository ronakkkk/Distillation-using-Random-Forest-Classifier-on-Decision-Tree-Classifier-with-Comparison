import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
encoder = LabelEncoder()
data = pandas.read_excel('Immunotherapy.xlsx')

print("************ Data Exploration **************")
print("Immunotherapy Data:\n", data.head())
print("\n Data Type:\n", data.dtypes)
print("\n Data Description:\n", data.describe())

# dropping nan rows
data = data.dropna()

# plot data
sns.barplot(x=data['Result_of_Treatment'], y=data['age'])
plt.show()
'''feature and label'''
X = data.loc[:, data.columns != 'Result_of_Treatment']
Y = data['Result_of_Treatment']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

'''Random Forest Classifier'''
print("************ Random Forest Classifier **************")
#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=500)

#Train the model using the training
clf.fit(X_train,y_train)
ypred=clf.predict(X_test)
print("Random Forest: ", accuracy_score(ypred, y_test))

'''Getting Best Hyperparameter using Grid Search'''
print("************ Best Hyperparameters for Decision Tree Classifier using Grid Search **************")
parameter = {
    'max_depth': [2,3,4,5,6,10,20,30],
    'min_samples_leaf': [5,10,15,20,25,30,35,40,45,50],
    'criterion': ['gini', 'entropy']
}
dtree_model = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator=dtree_model,
                           param_grid=parameter,
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)

'''Distillation Process'''
print("************ Decision Tree Classifier using Distillation process **************")
y_pred = clf.predict_proba(X)

# from sklearn import metrics
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
df_test = pandas.DataFrame()
df_test['prob_0'] = y_pred[:,0]
df_test['prob_1'] = y_pred[:,1]


# create bin using pandas
labels = ['0.0 - 0.2', '0.2 - 0.4', '0.4 - 0.6', '0.6 - 0.8', '0.8 - 1.0']
X['range_0'] = pandas.cut(df_test['prob_0'],labels=labels, bins=[0.0,0.2,0.4,0.6,0.8,1.0])
X['range_1'] = pandas.cut(df_test['prob_1'],labels=labels, bins=[0.0,0.2,0.4,0.6,0.8,1.0])
X = X.dropna()

# get df_bin
df_bin = pandas.DataFrame()
df_bin['range_0'] = X['range_0']
df_bin['range_1'] = X['range_1']

X = X.drop(['range_0', 'range_1'], axis=1)

print("Binned Probability:\n", df_bin.head())
xtrain, xtest, ytrain, ytest = train_test_split(X, df_bin, test_size=0.3)
dtree_model = DecisionTreeClassifier(max_depth=2, min_samples_split=2, min_samples_leaf=5, criterion='gini').fit(xtrain, ytrain)
dtree_predictions = dtree_model.predict(xtest)


ytest_lst = []
for i in ytest.values:
    if(i[0]>=i[1]):
        ytest_lst.append(i[0])

    else:
        ytest_lst.append(i[1])

dtree_predictions_lst =[]
for i in dtree_predictions:
    if (i[0] >= i[1]):
        dtree_predictions_lst.append(i[0])

    else:
        dtree_predictions_lst.append(i[1])

print("Decision Tree (Distillation): ", accuracy_score(dtree_predictions_lst, ytest_lst))

# tree plot
figure = plt.figure()
_ = sklearn.tree.plot_tree(dtree_model, feature_names = X.columns, class_names=df_bin.columns)
figure.savefig("immunotherapy_distillation_decision_tree.png")

'''Decision Tree'''
print("************ Decision Tree Classifier **************")
dtree_model = DecisionTreeClassifier(max_depth=2, min_samples_split=2, min_samples_leaf=5, criterion='gini').fit(X_train, y_train)
dtree_pred = dtree_model.predict(X_test)
print("Decision Tree: ", accuracy_score(dtree_pred, y_test))

# tree plot
figure = plt.figure()
_ = sklearn.tree.plot_tree(dtree_model, feature_names = X.columns, class_names=df_bin.columns)
figure.savefig("immunotherapy_decision_tree.png")