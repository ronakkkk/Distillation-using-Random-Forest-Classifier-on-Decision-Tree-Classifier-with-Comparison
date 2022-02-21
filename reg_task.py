import matplotlib.pyplot as plt
import pandas
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

encoder = LabelEncoder()
data = pandas.read_csv('forestfires.csv')

print("************ Data Exploration **************")
print("Forest Fire Data:\n", data.head())
print("\n Data Type:\n", data.dtypes)
print("\n Data Description:\n", data.describe())

# numeric conversion
data['month'] = encoder.fit_transform(data['month'])
data['day'] = encoder.fit_transform(data['day'])
data = data.dropna()

'''feature and label'''
X = data.loc[:, data.columns != 'area']
Y = data['area']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

'''Random Forest Regressor'''
print("************ Random Forest Regressor **************")
#Create a Gaussian Regressor
clf=RandomForestRegressor(n_estimators=500)

#Train the model using the training
clf.fit(X_train,y_train)
ypred=clf.predict(X_test)

print(mean_squared_error(ypred, y_test))


y_pred = clf.predict(X)



'''distillation Process'''
print("************ Decision Tree Classifier using Distillation process **************")
# create bin using pandas
labels = ['0.0', '15.0', '30.0', '45.0', '60.0']
X['range'] = pandas.cut(y_pred,labels=labels, bins=[0.0,15.0,30.0,45.0,60.0,75.0])
X = X.dropna()

# get df_bin
df_bin = pandas.DataFrame()
df_bin['range'] = X['range']

X = X.drop(['range'], axis=1)
print("Binned Probability:\n", df_bin.head())
xtrain, xtest, ytrain, ytest = train_test_split(X, df_bin, test_size=0.3)
dtree_model = DecisionTreeRegressor(max_depth=2).fit(xtrain, ytrain)
dtree_predictions = dtree_model.predict(xtest)

print(mean_squared_error(dtree_predictions, ytest))
# tree plot
figure = plt.figure()
_ = sklearn.tree.plot_tree(dtree_model, feature_names = X.columns, class_names=df_bin.columns)
figure.savefig("forestfires_distillation_decision_tree.png")

'''Decision Tree'''
print("************ Decision Tree Classifier **************")
dtree_model = DecisionTreeRegressor(max_depth=2).fit(X_train, y_train)
dtree_pred = dtree_model.predict(X_test)
print(mean_squared_error(dtree_pred, y_test))

# tree plot
figure = plt.figure()
_ = sklearn.tree.plot_tree(dtree_model, feature_names = X.columns, class_names=df_bin.columns)
figure.savefig("forestfires_decision_tree.png")



