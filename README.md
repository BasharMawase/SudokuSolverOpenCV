# SudokuSolver with OpenCV + MNIST / Решатель Sudoku на OpenCV и MNIST
## Table of Contents / Содержание
- [Overview / Обзор](#overview--обзор)
- [Project Structure / Структура проекта](#project-structure--структура-проекта)
- [Installation / Установка](#installation--установка)
- [Usage / Использование](#usage--использование)
- [Requirements / Требования](#requirements--требования)
  
## Overview / Обзор
**English**:  
A computer vision Sudoku solver that automatically:
1. Detects Sudoku grids using OpenCV
2. Recognizes digits with a pre-trained MNIST CNN model
3. Solves the puzzle using backtracking algorithm
4. Displays the solution on the original image

**Русский**:  
Программа для автоматического решения Sudoku, которая:
1. Находит сетку Sudoku на изображении (OpenCV)
2. Распознаёт цифры с помощью CNN-модели (MNIST)
3. Решает головоломку алгоритмом backtracking
4. Накладывает решение на исходное изображение
![telegram-cloud-photo-size-2-5280714333006982274-y](https://github.com/user-attachments/assets/6d08064a-5412-4c17-861e-662f0bc6fb35)
![telegram-cloud-photo-size-2-5274255883309807658-y](https://github.com/user-attachments/assets/9ced8717-42d3-40f1-92a6-3e4bc6cd709e)

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
## Installation / Установка
**English**:  
1. Clone the repository
2. Install dependencies

**Русский**:  
1. Склонируйте репозиторий
2. Установите зависимости

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

Advanced / Дополнительные возможности
English: Train the model (optional)
Русский: Обучить модель (опционально)

```bash
# English: Train for 100 epochs
# Русский: Обучение на 100 эпохах
python src/OCR_CNN_Training.py --epochs 10
Requirements / Требования
Core Dependencies / Основные зависимости:
```
```bash
opencv-python>=4.5  # Image processing / Обработка изображений
tensorflow>=2.10    # Machine learning / Машинное обучение
numpy>=1.23         # Matrix operations / Матричные операции
Optional / Опционально:
```
```bash
pytesseract         # OCR fallback / Резервное распознавание текста
matplotlib          # Visualization / Визуализация
```

### Контакты / Contacts:
- Email: basharmawaseru@gmail.com
- Issues: https://github.com/BasharMawase/SudokuSolverOpenCV
