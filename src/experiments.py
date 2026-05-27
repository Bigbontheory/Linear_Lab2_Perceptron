import numpy as np
import matplotlib.pyplot as plt
from dataset import prepare_all_data
from perceptron import Perceptron


def calc_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def print_table_header(title, param_name):
    print(f"\n{'=' * 45}")
    print(f"{title}")
    print(f"{'=' * 45}")
    print(f"{param_name:<15} | {'Train Acc':<12} | {'Test Acc':<12}")
    print("-" * 45)


def run_experiment_lr(X_train, y_train, X_test, y_test):
    lrs = [0.001, 0.01, 0.5, 1.0]
    plt.figure(figsize=(10, 6))

    print_table_header("Эксперимент 1: Влияние скорости обучения", "LR (eta)")

    for lr in lrs:
        model = Perceptron()
        model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=lr, batch_size=32, verbose=False)

        train_acc = calc_accuracy(y_train, model.predict(X_train)) * 100
        test_acc = calc_accuracy(y_test, model.predict(X_test)) * 100

        print(f"{lr:<15} | {train_acc:>6.2f}%{'':<5} | {test_acc:>6.2f}%")

        # Рисуем график loss на валидации (test)
        plt.plot(model.val_losses, label=f'LR = {lr}')

    plt.title('Влияние скорости обучения на Test Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss (BCE)')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_experiment_batch(X_train, y_train, X_test, y_test):
    batches = [1, 16, 64, 256]
    plt.figure(figsize=(10, 6))

    print_table_header("Эксперимент 2: Влияние размера батча", "Batch Size")

    for b_size in batches:
        model = Perceptron()
        model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=b_size, verbose=False)

        train_acc = calc_accuracy(y_train, model.predict(X_train)) * 100
        test_acc = calc_accuracy(y_test, model.predict(X_test)) * 100

        print(f"{b_size:<15} | {train_acc:>6.2f}%{'':<5} | {test_acc:>6.2f}%")

        plt.plot(model.val_losses, label=f'Batch = {b_size}')

    plt.title('Влияние размера батча на Test Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss (BCE)')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_experiment_init(X_train, y_train, X_test, y_test):
    modes = ["zero", "small", "large"]
    plt.figure(figsize=(10, 6))

    print_table_header("Эксперимент 3: Влияние инициализации весов", "Init Mode")

    for mode in modes:
        model = Perceptron()
        model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32, init_mode=mode, verbose=False)

        train_acc = calc_accuracy(y_train, model.predict(X_train)) * 100
        test_acc = calc_accuracy(y_test, model.predict(X_test)) * 100

        print(f"{mode:<15} | {train_acc:>6.2f}%{'':<5} | {test_acc:>6.2f}%")

        plt.plot(model.val_losses, label=f'Init = {mode}')

    plt.title('Влияние инициализации весов на Test Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss (BCE)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_all_data()

    print("Запуск экспериментов...")

    # Запускаем по очереди. Закрывай окно графика, чтобы начался следующий эксперимент.
    run_experiment_lr(X_train, y_train, X_test, y_test)
    run_experiment_batch(X_train, y_train, X_test, y_test)
    run_experiment_init(X_train, y_train, X_test, y_test)

    print("\nВсе эксперименты завершены!")