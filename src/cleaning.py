import pandas as pd

ages = [-1, 0, 5, 12, 18, 30, 60, 100]
labels = ['missing', 'infant', 'child', 'teen', 'young', 'adult', 'senior']
Title_dict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Mr": "Mr",
    "Ms": "Mrs",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}


def cleanName(name):
    w1 = name.split(",")[1]
    w2 = w1.split(".")[0].strip()
    return w2


def cleanTitanic(df, dropCols,  catClass, medianCols=[]):
    ''' Clean the dataframe of titanic drop irrelevant columns, 
    change to type category and fill few NaN values'''
    df['Age'].fillna(-0.5, inplace=True)
    df['Age'] = pd.cut(df['Age'], ages, labels=labels)
    df['Name'] = df['Name'].apply(lambda x: cleanName(x))
    df['Name'] = df.Name.map(Title_dict)
    df['SibSp'] = df['SibSp'].apply(lambda x: 2 if x > 1 else x)
    df['Parch'] = df['Parch'].apply(lambda x: 2 if x > 1 else x)
    df = df.drop(dropCols, axis=1)
    for col in medianCols:
        df[col].fillna(value=df[col].median(), inplace=True)
    for cat in catClass:
        df[cat] = df[cat].astype('category')
    df.Embarked.fillna(method="ffill", inplace=True)

    return df
