import numpy as np

# Parameters for the perceptron
num_inputs = 2
learning_rate = 0.1
epochs = 10

# Initialize weights and bias
weights = np.random.rand(num_inputs)
bias = 0

# Activation function (Step function using NumPy)
def activate(summation):
    return (summation >= 0).astype(int)

# Prediction function
def predict(inputs, weights, bias):
    summation = np.dot(inputs, weights) + bias
    return activate(summation)

# Training Data for AND Gate
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

labels = np.array([0, 0, 0, 1])

# Train the perceptron
print("Training Perceptron for AND gate\n")
for epoch in range(epochs):
    error_count = 0
    for inputs, label in zip(training_inputs, labels):
        prediction = predict(inputs, weights, bias)
        
        # Update weights and bias based on the error
        error = label - prediction
        if error != 0:
            error_count += 1
        weights += learning_rate * error * inputs
        bias += learning_rate * error
    
    # Print progress for each epoch
    print(f"Epoch {epoch + 1}/{epochs}: Weights={weights}, Bias={bias}, Errors={error_count}")
    # If no errors, perceptron has converged, stop training
    if error_count == 0:
        print(f"Converged after {epoch + 1} epochs.")
        break

print(f"\nFinal Weights: {weights}")
print(f"Final Bias: {bias}")
print("\nTraining complete.")

# Testing the Perceptron
print("\nTesting Perceptron, AND Gate Logic:")
test_cases = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

for inputs in test_cases:
    output = predict(inputs, weights, bias)
    print(f"Inputs: {inputs[0]}, {inputs[1]} -> Output: {output}")