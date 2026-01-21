# Implementing an AND Gate using a Simple Perceptron

import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Initialize weights to small random values and bias to 0
        self.weights = np.random.rand(num_inputs)
        self.bias = 0

    def activate(self, summation):
        # Step function activation
        return 1 if summation >= 0 else 0

    def predict(self, inputs):
        # Calculate the weighted sum of inputs and apply activation function
        summation = np.dot(inputs, self.weights) + self.bias
        return self.activate(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # Update weights and bias based on the error
                self.weights += self.learning_rate * (label - prediction) * inputs
                self.bias += self.learning_rate * (label - prediction)
        print(f"Final Weights: {self.weights}")
        print(f"Final Bias: {self.bias}")

# Define the training data for an AND gate
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

labels = np.array([0, 0, 0, 1])

# Create a perceptron with 2 inputs
perceptron = Perceptron(num_inputs=2, learning_rate=0.1, epochs=100)

# Train the perceptron
print("Training Perceptron for AND gate...")
perceptron.train(training_inputs, labels)
print("Training complete.")

# To test if the trained perceptron correctly implements the AND gate logic.

print("\nTesting Perceptron (AND Gate Logic):")

test_cases = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

for inputs in test_cases:
    output = perceptron.predict(inputs)
    print(f"Inputs: {inputs[0]}, {inputs[1]} -> Output: {output}")