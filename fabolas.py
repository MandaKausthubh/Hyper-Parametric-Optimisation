import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Define the search space for hyperparameters
search_space = {
    "learning_rate": (0.09, 0.1),  # Continuous space
    "num_layers": (1, 3),           # Discrete space
    "hidden_units": (1, 2),      # Discrete space
    "dataset_fraction": (0.5, 0.51)  # Continuous space
}

# Load MNIST dataset
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    return train_dataset, test_dataset

train_dataset, test_dataset = load_mnist()

# Define a simple neural network with flexible architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, num_layers):
        super(SimpleNN, self).__init__()
        layers = []
        in_features = input_size

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            in_features = hidden_units

        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Objective function for Fabolas
def objective_function(config):
    learning_rate = config["learning_rate"]
    num_layers = config["num_layers"]
    hidden_units = config["hidden_units"]
    dataset_fraction = config["dataset_fraction"]

    # Subsample the dataset based on the dataset_fraction
    num_samples = int(len(train_dataset) * dataset_fraction)
    subset_indices = np.random.choice(len(train_dataset), num_samples, replace=False)
    train_subset = Subset(train_dataset, subset_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN(28 * 28, 10, hidden_units, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model for a fixed number of epochs (e.g., 3 epochs for this demo)
    model.train()
    for epoch in range(3):
        for images, labels in train_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the full validation dataset
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return 1 - accuracy  # Minimize the error rate

# Gaussian Process for Fabolas
kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0, 1.0, 1.0], (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

def sample_configuration():
    return {
        "learning_rate": np.random.uniform(*search_space["learning_rate"]),
        "num_layers": np.random.randint(*search_space["num_layers"]),
        "hidden_units": np.random.randint(*search_space["hidden_units"]),
        "dataset_fraction": np.random.uniform(*search_space["dataset_fraction"]),
    }

# Fabolas optimization loop
def fabolas_optimize(iterations=10):
    observed_configs = []
    observed_losses = []

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")

        if len(observed_configs) < 5:
            config = sample_configuration()
        else:
            # Use the GP to predict the next best configuration
            X = np.array([[
                cfg["learning_rate"],
                cfg["num_layers"],
                cfg["hidden_units"],
                cfg["dataset_fraction"]
            ] for cfg in observed_configs])

            y = np.array(observed_losses)
            gp.fit(X, y)

            # Predict the next promising configuration (basic implementation)
            config = sample_configuration()

        # Evaluate the objective function
        loss = objective_function(config)
        observed_configs.append(config)
        observed_losses.append(loss)

        print(f"Iteration {i + 1}: Config: {config}, Loss: {loss:.4f}")

    # Return the best configuration
    best_index = np.argmin(observed_losses)
    best_config = observed_configs[best_index]
    best_loss = observed_losses[best_index]

    print(f"Best Config: {best_config}, Best Loss: {best_loss:.4f}")
    return best_config, best_loss

# Run Fabolas optimization
best_config, best_loss = fabolas_optimize(iterations=2)
