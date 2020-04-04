import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from cleaning import cleanTitanic
from CreateXandY import XandY

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

dropCols = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']

catClass = ['Pclass', 'Sex', 'Embarked', 'Age']

alive = train[train['Survived'] == 1]
deads = train[train['Survived'] == 0]

sns.boxplot(data=train, x="Survived", y='Fare')
plt.show()
print(train.columns)
#alive.Parch.plot.hist(alpha=.5, bins=18, color='red')
#deads.Parch.plot.hist(alpha=.5, bins=18, color='blue')
#plt.legend(['alive', 'dead'])
# plt.show()

testCleaned = cleanTitanic(test, dropCols, catClass, ['Fare'])
trainCleaned = cleanTitanic(train, dropCols, catClass=catClass)

X, y = XandY(trainCleaned, dummyCols=catClass)
print(X.columns)
testCleaned['Survived'] = [None]*testCleaned.shape[0]
Xresult, yresult = XandY(testCleaned, dummyCols=catClass)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(Xtrain, ytrain)

ypred = model.predict(Xtest)
print("Accuracy with non seen data: ", accuracy_score(ypred, ytest))
scores = cross_val_score(model, X, y, cv=5)

print("Score with cross validation: ", scores.mean())

test['Survived'] = model.predict(Xresult)

test[['PassengerId', "Survived"]].to_csv('kaggle_submission3.csv', index=False)
