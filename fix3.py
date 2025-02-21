import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load or create your RandomForestClassifier model
iris = load_iris()
X_train, y_train = iris.data, iris.target

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model/random_forest.pkl')
