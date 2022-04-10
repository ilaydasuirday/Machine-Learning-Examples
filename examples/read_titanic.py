import pandas as pd
import csv
titanic_df = pd.read_csv("titanic.csv", sep = ",")
print(titanic_df)
print(titanic_df.head(10))
print(titanic_df.tail(10))
print(type(titanic_df))
print(titanic_df.dtypes)

titanic_df.shape
n_row = titanic_df.shape[0]
print(n_row)
n_col = titanic_df.shape[1]
print(n_col)
print("row, column", n_row, n_col)

titanic_df.info()
titanic_df.describe()

titanic_df['Pclass'].unique()
titanic_df.Pclass.unique()
titanic_df.iloc[:, 2].unique()

titanic_df['Sex'].value_counts()

first_hunderd = titanic_df.iloc[0:100, :]
print(first_hunderd)

first_hunderd= titanic_df.loc[0:99, :]
print(first_hunderd)

names = titanic_df.pop('Name')
print(names)

first_six_col = titanic_df.iloc[0:6]
print(first_six_col)

col_list = ['Age', 'Sex', 'Survived']
new = titanic_df.loc[0: 99, col_list]
print(new)

survived = titanic_df[titanic_df['Survived'] == 1]
print(survived)

survived['Sex'].value_counts()

survived_men = titanic_df[(titanic_df['Survived'] == 1) & (titanic_df['Sex'] == 'male')]
print(survived_men)

male_age_avg = survived_men['Age'].mean()
print(male_age_avg)

print(titanic_df.isnull())
print(titanic_df.isnull().sum(axis=0))

titanic_df.dropna(inplace =True)

titanic_df.shape
print(titanic_df.corr())






















