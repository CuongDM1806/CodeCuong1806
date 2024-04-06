import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
# Load the CSV file into a DataFrame
df = pd.read_csv("Dry_Bean_Dataset.csv")
X = df.iloc[:, 1:-1].values
y_str = df.iloc[:, -1].values
class_mapping = {'SEKER': 0, 'BARBUNYA': 1, 'BOMBAY': 2, 'CALI': 3, 'HOROZ': 4, 'SIRA': 5, 'DERMASON': 6}

# Map class strings to numbers
y = np.array([class_mapping[class_str] for class_str in y_str])
# Perform PCA to reduce dimensionality
pca = PCA(n_components=2)  # Reduce to 3 dimensions for visualization
X_reduced = pca.fit_transform(X)

# Create a DataFrame for visualization
df_reduced = pd.DataFrame(data=X_reduced, columns=['PC1', 'PC2'])
df_reduced['Class'] = y

# Visualize in 2D
plt.figure(figsize=(10, 6))
plt.scatter(df_reduced['PC1'], df_reduced['PC2'], c=df_reduced['Class'], cmap='viridis')
plt.title('PCA Visualization (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Bean Type')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import numpy as np

# Tính toán số lượng phần tử của mỗi class trong y_train
class_counts = np.bincount(y_train)

# In ra số lượng phần tử của mỗi class
for class_index, count in enumerate(class_counts):
    class_name = list(class_mapping.keys())[list(class_mapping.values()).index(class_index)]
    print(f"Class {class_name}: {count} instances")

# Initialize and train the Naïve Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = nb_classifier.predict(X_test)
print(y_pred)
# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print("Naïve Bayes Classifier Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

from sklearn.linear_model import LogisticRegression

# Initialize and train the Multinomial Logistic Regression classifier
logistic_classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
logistic_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred_logistic = logistic_classifier.predict(X_test)

# Calculate accuracy, precision, and recall
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic, average='macro')
recall_logistic = recall_score(y_test, y_pred_logistic, average='macro')

print("Multinomial Logistic Regression Performance:")
print("Accuracy:", accuracy_logistic)
print("Precision:", precision_logistic)
print("Recall:", recall_logistic)

#ANN method
def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    exp_scores = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

def cost(Y, Yhat):
    """
    Compute the cross-entropy loss.
    """
    return -np.mean(np.sum(Y * np.log(Yhat), axis=0))

from scipy import sparse
def convert_labels(y, C = 7):
    Y = sparse.coo_matrix((np.ones_like(y),
    (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y


N, d0 = X.shape
print(N, d0)
d1 = h = 100  # Number of neurons in hidden layer
d2 = C = 7   # Number of classes

# Initialize parameters randomly
W1 = 0.01 * np.random.randn(d0, d1)
b1 = np.zeros((d1, 1))
W2 = 0.01 * np.random.randn(d1, d2)
b2 = np.zeros((d2, 1))

Y = convert_labels(y_train, C)
eta = 1 # Learning rate

for i in range(10000):
    # Feedforward
    Z1 = np.dot(W1.T, X_train.T) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    Yhat = softmax(Z2)
    
    # Print loss after each 1000 iterations
    if i % 1000 == 0:
        # Compute the loss: average cross-entropy loss
        loss = cost(Y, Yhat)
        print("iter %d, loss: %f" % (i, loss))

    # Backpropagation
    E2 = (Yhat - Y) / N
    dW2 = np.dot(A1, E2.T)
    db2 = np.sum(E2, axis=1, keepdims=True)
    E1 = np.dot(W2, E2)
    E1[Z1 <= 0] = 0  # Gradient of ReLU
    dW1 = np.dot(X_train.T, E1.T)
    db1 = np.sum(E1, axis=1, keepdims=True)

    # Gradient Descent update
    W1 += -eta * dW1
    b1 += -eta * db1
    W2 += -eta * dW2
    b2 += -eta * db2

# Evaluate training accuracy
Z1 = np.dot(W1.T,X_test.T ) + b1
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
predicted_class = np.argmax(Z2, axis=0)
print(predicted_class)
acc = 100 * np.mean(predicted_class == y_test)
print('Training accuracy with ANN method: %.2f %%' % acc)

