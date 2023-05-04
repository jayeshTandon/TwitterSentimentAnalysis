#%%
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

#%%
# Load pre-processed data
df = pd.read_csv('preProcessed.csv')

#%%
# Create dataset
dataset = pd.DataFrame()
dataset['text'] = df['text']
dataset['target'] = df['target']

#%%
# Vectorize text data
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(dataset['text'].astype(str)).toarray()
y = dataset['target'].values

#%%
# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

#%%
# Fit a Random Forest classifier to the training data
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(x_train, y_train)

#%%
# Make predictions on the testing data
y_pred = classifier.predict(x_test)

#%%
# Print confusion matrix and accuracy score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

#%%
# Save the trained model and CountVectorizer for later use
with open('model_rf.pkl', 'wb') as f:
    pickle.dump((classifier, cv), f)

# %%
