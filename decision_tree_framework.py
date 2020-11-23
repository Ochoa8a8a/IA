import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# Load training data frame
df = pd.read_csv('train.csv')
df.replace(['male', 'female', 'S', 'C', 'Q'], [0, 1, 0, 1, 2], inplace=True)
df.fillna(0, inplace=True)
# Load testing data frame
df_t = pd.read_csv('test.csv')
df_t.replace(['male', 'female', 'S', 'C', 'Q'], [0, 1, 0, 1, 2], inplace=True)
df_t.fillna(0, inplace=True)
# Create a list of the feature column's names
features = df.columns[2:8]
print(features)

y = df['Survived']

# Create a random forest Classifier.
clf = DecisionTreeClassifier(criterion="entropy")
# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(df[features], y)

# Create predictions
preds = clf.predict(df_t[features])
dr = pd.DataFrame(preds, columns=['Survived'])
dr.to_csv('results_framework.csv')
