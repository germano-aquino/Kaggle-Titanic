import pandas as pd

ages = [-1, 0, 5, 12, 18, 30, 60, 100]
labels = ['missing', 'infant', 'child', 'teen', 'young', 'adult', 'senior']


def cleanTitanic(df, dropCols,  catClass, medianCols=[]):
    ''' Clean the dataframe of titanic drop irrelevant columns, 
    change to type category and fill few NaN values'''
    df['Age'].fillna(-0.5, inplace=True)
    df['Age'] = pd.cut(df['Age'], ages, labels=labels)
    df['SibSp'] = df['SibSp'].apply(lambda x: 2 if x > 1 else x)
    df['Parch'] = df['Parch'].apply(lambda x: 2 if x > 1 else x)
    df = df.drop(dropCols, axis=1)
    for col in medianCols:
        df[col].fillna(value=df[col].median(), inplace=True)
    for cat in catClass:
        df[cat] = df[cat].astype('category')
    df.Embarked.fillna(method="ffill", inplace=True)

    return df
