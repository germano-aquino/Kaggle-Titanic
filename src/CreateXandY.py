import pandas as pd


def XandY(df, dummyCols):
    '''Transform dataframe into X and y inputs on ML models'''
    temp = pd.get_dummies(df[dummyCols], prefix_sep='_', drop_first=True)
    y = df['Survived']
    dummyCols = dummyCols + ['PassengerId', 'Survived']
    df.drop(columns=dummyCols, inplace=True)
    X = pd.concat([df, temp], axis=1)
    return X, y
