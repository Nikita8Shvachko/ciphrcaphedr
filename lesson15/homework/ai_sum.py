### ЗАДАЧА НА ПРАКТИКУ № 2 (со звездочкой)
# Написать нейросеть, которая будет складывать два небольших числа (от 0 до 10)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class SumAi(nn.Module):
    def __init__(self):
        super(SumAi, self).__init__()
        self.layer1 = nn.Linear(2, 10)  # Вход: 2 числа -> 10 нейронов
        self.layer2 = nn.Linear(10, 5)  # Скрытый слой: 10 -> 5 нейронов
        self.layer3 = nn.Linear(5, 1)  # Выход: 5 -> 1 число (сумма)
        self.relu = nn.ReLU()  # Активационная функция

    def forward(self, x):  # прямой проход
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# создание модели
model = SumAi()
criterion = nn.MSELoss()  # среднеквадратическая ошибка (MSE)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer


# генерация данных
def generate_data(n_samples=1000):
    # генерация случайных пар чисел между 0 и 10
    x = np.random.uniform(0, 10, size=(n_samples, 2))
    # вычисление суммы
    y = x.sum(axis=1, keepdims=True)
    return torch.FloatTensor(x), torch.FloatTensor(y)


# цикл обучения
def train_model(model, epochs=500):
    # Создаем прогресс-бар
    pbar = tqdm(range(epochs), desc="Обучение", unit="эпох")

    for epoch in pbar:
        # генерация новых данных
        inputs, targets = generate_data(100)

        # прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # обратный проход и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Обновляем прогресс-бар с текущей ошибкой
        pbar.set_postfix({"Ошибка": f"{loss.item():.4f}"})


def test_precision(model, num_tests=1000):
    """Тестирование точности модели с разным количеством знаков после запятой"""
    print("\nТестирование точности модели:")
    print("-" * 50)

    # Тестируем разные количества знаков после запятой
    for decimals in range(4):  # от 0 до 3 знаков после запятой
        errors = []
        for _ in range(num_tests):
            # Генерируем числа с заданным количеством знаков после запятой
            num1 = round(np.random.uniform(0, 10), decimals)
            num2 = round(np.random.uniform(0, 10), decimals)

            # Получаем предсказание модели
            input_tensor = torch.FloatTensor([[num1, num2]])
            predicted_sum = model(input_tensor).item()
            actual_sum = num1 + num2

            # Вычисляем ошибку
            error = abs(predicted_sum - actual_sum)
            errors.append(error)

        # Выводим статистику
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        print(f"Точность для {decimals} знаков после запятой:")
        print(f"Средняя ошибка: {avg_error:.6f}")
        print(f"Максимальная ошибка: {max_error:.6f}")
        print("-" * 50)


# обучение модели
def main():
    while True:
        print("\n=== Нейронная сеть для сложения чисел ===")
        print("1. Обучить модель")
        print("2. Протестировать модель")
        print("3. Проверить точность")
        print("4. Выйти")

        choice = input("\nВыберите действие (1-4): ")

        if choice == "1":
            try:
                epochs = int(input("Введите количество эпох обучения (100-50000): "))
                if not 100 <= epochs <= 50000:
                    print("Количество эпох должно быть между 100 и 50000.")
                    continue
                print("\nОбучение модели...")
                train_model(model, epochs)
                print("\nОбучение завершено!")
            except ValueError:
                print("Введите корректное количество эпох.")

        elif choice == "2":
            test_model(model)

        elif choice == "3":
            test_precision(model)

        elif choice == "4":
            print("До свидания!")
            break

        else:
            print("Некорректный выбор! Введите 1, 2, 3 или 4.")


# тестирование модели
def test_model(model):
    while True:
        try:
            num1 = float(input("\nВведите первое число (0-10): "))
            num2 = float(input("Введите второе число (0-10): "))

            if not (0 <= num1 <= 10 and 0 <= num2 <= 10):
                print("Числа должны быть между 0 и 10.")
                continue

            input_tensor = torch.FloatTensor([[num1, num2]])
            predicted_sum = model(input_tensor).item()
            actual_sum = num1 + num2

            print(f"Предсказанная сумма: {predicted_sum:.6f}")
            print(f"Фактическая сумма: {actual_sum:.6f}")
            print(f"Ошибка: {abs(predicted_sum - actual_sum):.6f}")

            if input("\nПротестировать другую пару? (y/n): ").lower() != "y":
                break

        except ValueError:
            print("Введите корректные числа.")


if __name__ == "__main__":
    main()
