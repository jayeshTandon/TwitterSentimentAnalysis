#%%
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

#%%
df = pd.read_csv('preProcessed.csv')

#%%
dataset = pd.DataFrame()
dataset['text'] = df['text']
dataset['target'] = df['target']

#%%
cv = CountVectorizer(max_features=1000)

array = dataset['text']
array = array.astype(str)

#%%
X = cv.fit_transform(array).toarray()
y = dataset.iloc[:,-1].values

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=14)

#%%
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# %%
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# %%
with open('model_nb.pkl', 'wb') as f:
    pickle.dump((classifier,cv) , f)