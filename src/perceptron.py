import numpy as np


class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None
        self.train_losses = []
        self.val_losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32, init_mode="small", verbose=True):
        n_samples, n_features = X_train.shape

        if init_mode == "small":
            self.w = np.random.randn(n_features) * 0.01
        elif init_mode == "zero":
            self.w = np.zeros(n_features)
        elif init_mode == "large":
            self.w = np.random.randn(n_features) * 10

        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_train_shuffled = X_train[idx]
            y_train_shuffled = y_train[idx]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                y_pred = self.forward(X_batch)
                error = y_pred - y_batch
                m = X_batch.shape[0]

                grad_w = (X_batch.T.dot(error)) / m
                grad_b = np.mean(error)

                self.w -= lr * grad_w
                self.b -= lr * grad_b

            train_pred = self.forward(X_train)
            train_loss = self.compute_loss(y_train, train_pred)
            self.train_losses.append(train_loss)

            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                self.val_losses.append(val_loss)
            else:
                val_loss = 0.0

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)