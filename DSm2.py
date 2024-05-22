import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
train_df = pd.read_csv('train.csv')

# Handle missing values
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Drop the 'Cabin' column due to too many missing values
train_df = train_df.drop(columns=['Cabin'])

# Convert 'Sex' and 'Embarked' to categorical types and then to numeric for correlation
train_df['Sex'] = train_df['Sex'].astype('category').cat.codes
train_df['Embarked'] = train_df['Embarked'].astype('category').cat.codes

# Drop columns that are not useful for analysis
train_df = train_df.drop(columns=['PassengerId', 'Name', 'Ticket'])

# Summary statistics
print(train_df.describe(include='all'))

# Visualization
# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Gender')
plt.show()

# Survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Age distribution of survivors vs non-survivors
sns.histplot(train_df[train_df['Survived'] == 1]['Age'], bins=30, kde=False, color='green', label='Survived')
sns.histplot(train_df[train_df['Survived'] == 0]['Age'], bins=30, kde=False, color='red', label='Did not survive')
plt.legend()
plt.title('Age Distribution of Survivors vs Non-survivors')
plt.show()

# Correlation matrix
corr_matrix = train_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()