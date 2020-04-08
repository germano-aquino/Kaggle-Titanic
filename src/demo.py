from CreateXandY import XandY
from cleaning import cleanTitanic

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = 100


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

dropCols = ['Ticket']

catClass = ['Pclass', 'Sex', 'Embarked',
            'Age', 'Name', 'SibSp', 'Parch', 'Cabin']

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
trainCleaned = cleanTitanic(
    train, dropCols, catClass=catClass, medianCols=['Fare'])

# print(trainCleaned.head())
#sns.catplot(kind='bar', data=trainCleaned, x='Name', y='Survived', hue='Sex')
# plt.show()
testCleaned[testCleaned.Name.isnull()]['Name'] = 'Dona'

print(testCleaned.head())


X, y = XandY(trainCleaned, dummyCols=catClass)
testCleaned['Survived'] = [None]*testCleaned.shape[0]
Xresult, yresult = XandY(testCleaned, dummyCols=catClass)
Xresult['Name_Royalty'] = [0]*Xresult.shape[0]
Xresult['Cabin_T'] = [0]*Xresult.shape[0]


print("Formato de X ", X.head())
print("Formato de Xresult ", Xresult.head())

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

scores = cross_val_score(model, X, y, cv=5)
print("Score with cross validation: ", scores.mean())

test['Survived'] = model.predict(Xresult)

test[['PassengerId', "Survived"]].to_csv('kaggle_submission5.csv', index=False)

features = pd.DataFrame()
features['features'] = X.columns
features['importance'] = model.feature_importances_
features.sort_values('importance', inplace=True, ascending=False)
sns.catplot(data=features, kind='bar', y='features',
            x='importance', orient='h')
plt.show()
