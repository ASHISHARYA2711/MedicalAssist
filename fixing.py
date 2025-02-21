import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

# Load iris dataset for demonstration purposes
iris = load_iris()
X, y = iris.data, iris.target

# Train a simple Naive Bayes model (replace this with your actual model training code)
model_nb = GaussianNB()
model_nb.fit(X, y)

# Save the trained Naive Bayes model
joblib.dump(model_nb, 'model/naive_bayes.pkl')
