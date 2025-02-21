import joblib
import numpy as np

# Load the model
clf = joblib.load('model/decision_tree.pkl')
# Get the expected number of features
expected_num_features = clf.n_features_in_

# Your symptoms list
symptoms = ['abdominal_pain', 'back_pain', 'chills', 'nausea', 'dizziness']

# Create a list of zeros with the expected number of features
input_features = [0] * expected_num_features

# Print the number of expected features
print(f"Expected Number of Features: {expected_num_features}")

# Print the input features
print(f"Input Features: {input_features}")

# Convert the input features to a NumPy array
test = np.array(input_features).reshape(1, -1)

# Print the shape of the input array
print(f"Input Array Shape: {test.shape}")

# M
