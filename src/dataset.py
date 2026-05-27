import numpy as np
from sklearn.datasets import make_classification


def generate_data():
    return make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )


def stratified_train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)

    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)

    n_test_0 = int(len(class_0_indices) * test_size)
    n_test_1 = int(len(class_1_indices) * test_size)

    test_idx_0 = class_0_indices[:n_test_0]
    train_idx_0 = class_0_indices[n_test_0:]

    test_idx_1 = class_1_indices[:n_test_1]
    train_idx_1 = class_1_indices[n_test_1:]

    train_indices = np.concatenate([train_idx_0, train_idx_1])
    test_indices = np.concatenate([test_idx_0, test_idx_1])

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1e-8

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled


def prepare_all_data():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test