import pandas as pd
import numpy as np
# import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
# Load the data
transactions = pd.read_csv('transactions.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
print('----------------COUNT------------------------')
print(transactions['isFraud'].value_counts( ))

# Summary statistics on amount column
print('----------------------------------------')
print(transactions['amount'].describe())

# Create isPayment field
transactions['isPayment'] = np.where((transactions['type'] == 'PAYMENT') | (transactions['type'] == 'DEBIT'), 1, 0)

# Create isMovement field 1 when type is either “CASH_OUT” or “TRANSFER”, and a 0 otherwise.
transactions['isMovement'] = np.where((transactions['type'] == 'CASH_OUT') | (transactions['type'] == 'TRANSFER'), 1, 0)

# Create accountDiff field with the absolute difference of the oldbalanceOrg and oldbalanceDest columns.
transactions['accountDiff'] = abs(transactions['oldbalanceOrg'] - transactions['oldbalanceDest'])

# Create features and label variables
label = transactions['isFraud']
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)

# Normalize the features variables
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

# Fit the model to the training data
model = LogisticRegression()
model.fit(train_features, y_train)
# Score the model on the training data
score = model.score(train_features, y_train)
print('----------------------------------------')
print(score)

# Score the model on the test data
score_test = model.score(test_features, y_test)
print('----------------------------------------')
print(score_test)

# Print the model coefficients
coef = model.coef_
print('----------------------------------------')
print(coef)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction = np.array([6472.54, 1.0, 0.0, 55901.23])

# Combine new transactions into a single array
sample_transactions = np.stack([transaction1, transaction2, transaction3, your_transaction])

# Normalize the new transactions
sample_transactions_scaled = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
sample_transactions_fraud = model.predict(sample_transactions_scaled)
print('----------------------------------------')
print(sample_transactions_fraud)
# Show probabilities on the new transactions
sample_transactions_prob = model.predict_proba(sample_transactions_scaled)
print('----------------------------------------')
print(sample_transactions_prob)