import numpy as np
import matplotlib.pyplot as plt
from dataset import prepare_all_data
from perceptron import Perceptron


def calculate_accuracy(y_true, y_pred):
    """Вычисляет долю правильных ответов (accuracy)"""
    return np.mean(y_true == y_pred)


def main():
    # Загружаем подготовленные данные из нашего dataset.py
    X_train, X_test, y_train, y_test = prepare_all_data()

    # Инициализируем модель
    model = Perceptron()

    print("Начинаем обучение перцептрона...")
    # 3.1 Обучение с параметрами: lr=0.1, epochs=100, batch_size=32
    # В качестве валидационной выборки передаем тестовую (X_test, y_test)
    model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32, verbose=False)
    print("Обучение завершено!\n")

    # 3.3 Вычисление точности (accuracy) на обучающей и тестовой выборках
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = calculate_accuracy(y_train, train_preds)
    test_acc = calculate_accuracy(y_test, test_preds)

    print(f"Точность на обучающей выборке (Train Accuracy): {train_acc * 100:.2f}%")
    print(f"Точность на тестовой выборке (Test Accuracy): {test_acc * 100:.2f}%")

    # Настройка окна для графиков
    plt.figure(figsize=(12, 5))

    # 3.2 График изменения функции потерь
    plt.subplot(1, 2, 1)
    plt.plot(model.train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(model.val_losses, label='Test (Val) Loss', color='orange', linewidth=2, linestyle='--')
    plt.title('Изменение функции потерь (BCE)')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 3.4 Визуализация разделяющей границы на тестовых данных
    plt.subplot(1, 2, 2)

    # Рисуем точки тестовой выборки
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='red', label='Класс 0', alpha=0.7)
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='blue', label='Класс 1', alpha=0.7)

    # Уравнение разделяющей прямой: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
    # Находим крайние точки по оси X, чтобы провести линию от края до края графика
    x1_min, x1_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5

    # Вычисляем соответствующие значения по оси Y (x2)
    x2_min = -(model.w[0] * x1_min + model.b) / model.w[1]
    x2_max = -(model.w[0] * x1_max + model.b) / model.w[1]

    # Рисуем саму прямую
    plt.plot([x1_min, x1_max], [x2_min, x2_max], 'k--', linewidth=2, label='Разделяющая граница')

    plt.title('Разделяющая граница (Test Data)')
    plt.xlabel('Признак 1 (z-score)')
    plt.ylabel('Признак 2 (z-score)')
    plt.xlim(x1_min, x1_max)
    plt.ylim(X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5)
    plt.legend()
    plt.grid(True)

    # Отрисовываем графики на экране
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()