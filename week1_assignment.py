"""Week 1 ML assignment and Titanic data cleaning.
Put Kaggle Titanic train.csv in the same folder, then run: python week1_assignment.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DATA = Path(__file__).with_name('train.csv')
if not DATA.exists():
    raise FileNotFoundError('Download Kaggle Titanic train.csv and put it next to this script.')
df = pd.read_csv(DATA)
required = {'Survived', 'Age', 'Fare', 'Sex', 'Embarked'}
if not required.issubset(df.columns):
    raise ValueError(f'Missing columns: {required - set(df.columns)}')
print('First 10 rows:\n', df.head(10))
print('\nDataset information:')
df.info()
print('\nSummary statistics:\n', df.describe(include='all').to_string())
print('\nMissing values before cleaning:\n', df.isna().sum().to_string())

# Mean and median imputation: Fare is numerical; Age is skewed, so use the median.
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])
if 'Cabin' in df:
    df['Cabin'] = df['Cabin'].fillna('Unknown')
for col in df.select_dtypes(include='number').columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include='object').columns:
    if df[col].isna().any():
        df[col] = df[col].fillna('Unknown')

# LabelEncoder and OneHotEncoder are demonstrated on different columns.
df['Sex_encoded'] = LabelEncoder().fit_transform(df['Sex'])
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded = encoder.fit_transform(df[['Embarked']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Embarked']), index=df.index).astype(int)
df = pd.concat([df, encoded_df], axis=1)

plt.figure(figsize=(8, 5))
plt.hist(df['Age'], bins=25, edgecolor='black')
plt.title('Titanic Passenger Age Distribution')
plt.xlabel('Age (years)')
plt.ylabel('Passengers')
plt.tight_layout()
plt.savefig(DATA.with_name('age_distribution.png'), dpi=160)
plt.close()
output = DATA.with_name('titanic_cleaned.csv')
df.to_csv(output, index=False)
print('\nMissing values after cleaning:\n', df.isna().sum().to_string())
print('\nSaved:', output.name, 'and age_distribution.png')
