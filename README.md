# Sudoku Solver with OpenCV & MNIST / Решатель Sudoku на OpenCV и MNIST

<div align="center">
  <img src="https://github.com/user-attachments/assets/6d08064a-5412-4c17-861e-662f0bc6fb35" width="45%">
  <img src="https://github.com/user-attachments/assets/9ced8717-42d3-40f1-92a6-3e4bc6cd709e" width="45%">
</div>

## Table of Contents / Содержание
- [Overview](#overview--обзор)
- [Features](#features--возможности)
- [Project Structure](#project-structure--структура-проекта)
- [Installation](#installation--установка)
- [Usage](#usage--использование)
- [Advanced](#advanced--дополнительные-возможности)
- [Requirements](#requirements--требования)
- [Contacts](#contacts--контакты)

---

## Overview / Обзор
**English**:  
An intelligent Sudoku solver that combines computer vision and machine learning to:
1. Detect Sudoku grids in images using OpenCV
2. Recognize digits with a CNN model trained on MNIST
3. Solve puzzles using an optimized backtracking algorithm
4. Overlay the solution seamlessly on the original image

**Русский**:  
Программа для автоматического решения Sudoku, которая:
1. Обнаруживает сетку Sudoku на изображениях (OpenCV)
2. Распознаёт цифры с помощью CNN-модели (обученной на MNIST)
3. Решает головоломку с использованием алгоритма backtracking
4. Накладывает решение на исходное изображение

---

## Features / Возможности
- **Image Processing** / Обработка изображений
  - Perspective correction / Коррекция перспективы
  - Grid detection / Обнаружение сетки
- **Digit Recognition** / Распознавание цифр
  - 98.7% accuracy on MNIST / Точность 98.7% на MNIST
  - Custom CNN architecture / Собственная архитектура CNN
- **Sudoku Solving** / Решение Sudoku
  - Backtracking algorithm / Алгоритм backtracking
  - Real-time visualization / Визуализация в реальном времени

---
## Project Structure / Структура проекта
```
SudokuSolver/
├── src/
│   ├── SudokuSolver.py         # Main solver logic / Основной алгоритм
│   ├── OCR_CNN_Training.py     # Model training / Обучение модели
│   ├── OCR_CNN_Test.py         # Model testing / Тестирование модели
│   └── utils.py                # Helper functions / Вспомогательные функции
├── models/
│   ├── digit_classifier.keras  # Pretrained model / Готовая модель
│   └── best_model.keras        # Best model / Лучшая версия модели
├── data/
│   └── examples/               # Sample puzzles / Примеры изображений
└── requirements.txt            # Dependencies / Зависимости
```
---

---

## Installation / Установка
```bash
# Clone repository / Клонировать репозиторий
git clone https://github.com/BasharMawase/SudokuSolverOpenCV.git
cd SudokuSolver

# Install dependencies / Установить зависимости
pip install -r requirements.txt
```

```bash
git clone https://github.com/BasharMawase/SudokuSolverOpenCV.git
cd SudokuSolver
pip install -r requirements.txt
```
Usage / Использование

English: Solve a Sudoku from image
Русский: Решить Sudoku с изображения
```bash
python src/SudokuSolver.py --image data/examples/sudoku1.jpg
```
---
Options / Параметры:
```bash
--image      Path to Sudoku image / Путь к изображению
--debug      Show processing steps / Показать этапы обработки
--save       Save solution image / Сохранить решение
```
---
Advanced / Дополнительные возможности
English: Train the model (optional)
Русский: Обучить модель (опционально)

```bash
# English: Train for 100 epochs
# Русский: Обучение на 100 эпохах
python src/OCR_CNN_Training.py --epochs 100 --batch_size 32
```
---
## Requirements / Требования
```bash
opencv-python>=4.5  # Image processing / Обработка изображений
tensorflow>=2.10    # Machine learning / Машинное обучение
numpy>=1.23         # Matrix operations / Матричные операции
```
```bash
matplotlib          # Visualization / Визуализация
```

### Контакты / Contacts:
- Email: basharmawaseru@gmail.com
- Issues: https://github.com/BasharMawase/SudokuSolverOpenCV
