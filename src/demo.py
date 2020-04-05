import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from cleaning import cleanTitanic
from CreateXandY import XandY

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

dropCols = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']

catClass = ['Pclass', 'Sex', 'Embarked', 'Age']

alive = train[train['Survived'] == 1]
deads = train[train['Survived'] == 0]

#sns.boxplot(data=train, x="Survived", y='Fare')
# plt.show()
# print(train.columns)
#alive.Age.plot.hist(alpha=.5, bins=50, color='red')
#deads.Age.plot.hist(alpha=.5, bins=50, color='blue')
#plt.legend(['alive', 'dead'])
# plt.show()

testCleaned = cleanTitanic(test, dropCols, catClass, ['Fare'])
trainCleaned = cleanTitanic(train, dropCols, catClass=catClass)

X, y = XandY(trainCleaned, dummyCols=catClass)
print(X.Fare.head())
testCleaned['Survived'] = [None]*testCleaned.shape[0]
Xresult, yresult = XandY(testCleaned, dummyCols=catClass)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
params_rf = {'n_estimators': [25, 30, 35], 'max_features': [
    'log2', 'auto', 'sqrt'], 'min_samples_leaf': [3, 4, 5]}
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf,
                       cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(Xtrain, ytrain)

model = grid_rf.best_estimator_
print(grid_rf.best_score_)
print(grid_rf.best_params_)
ypred = model.predict(Xtest)

print("Accuracy with non seen data: ", accuracy_score(ypred, ytest))
#scores = cross_val_score(model, X, y, cv=5)

#print("Score with cross validation: ", scores.mean())

test['Survived'] = model.predict(Xresult)

test[['PassengerId', "Survived"]].to_csv('kaggle_submission4.csv', index=False)
