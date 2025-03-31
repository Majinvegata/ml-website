import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Load dataset
msg = pd.read_csv('naivetext.csv', names=['message', 'label'])
print('The dimensions of the dataset', msg.shape)
# Convert labels into numerical format
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
X = msg.message
y = msg.labelnum
# Splitting dataset into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
print('\nThe total number of Training Data:', ytrain.shape)
print('\nThe total number of Test Data:', ytest.shape)
# Convert text data into numerical feature vectors
cv = CountVectorizer()
xtrain_dtm = cv.fit_transform(xtrain)
xtest_dtm = cv.transform(xtest)
print('\nThe words or Tokens in the text documents \n')
print(cv.get_feature_names_out())  # Updated function for newer sklearn versions
# Convert to DataFrame (optional for debugging)
df = pd.DataFrame(xtrain_dtm.toarray(), columns=cv.get_feature_names_out())
# Train Naive Bayes classifier
clf = MultinomialNB().fit(xtrain_dtm, ytrain)
# Predict using the trained model
predicted = clf.predict(xtest_dtm)
# Printing evaluation metrics
print('\nAccuracy of the classifier is:', metrics.accuracy_score(ytest, predicted))
print('\nConfusion matrix:')
print(metrics.confusion_matrix(ytest, predicted))
print('\nThe value of Precision:', metrics.precision_score(ytest, predicted))
print('\nThe value of Recall:', metrics.recall_score(ytest, predicted))