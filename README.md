# Обнаружение и отслеживание курения по видеопотоку в реальном времени

ДЗ №28 по ТММ 2026 — Python (YOLO26n) + CoreML

---

## Как это работает

1. **YOLO26n** детектирует два класса в каждом кадре: `person` и `smoke`
2. Если bbox класса `smoke` пересекается с bbox класса `person` → факт курения
3. Модель обучена на датасете [Smoking Person Detection](https://universe.roboflow.com/project-i6bzi/smoking-person-detection-ec7ec)

---

## Структура проекта

```
Training/
├── train.py                 # обучение YOLO26n
├── export_coreml.py         # экспорт best.pt → .mlpackage для iOS
├── check_coreml_detections.py  # проверка выходов модели на тестовом изображении
├── data.yaml                # конфиг датасета (nc=2: person, smoke)
├── requirements.txt
└── datasets/
    └── smoking_detection/   # датасет с курящими людьми
```

---

## Результаты обучения

| Метрика    | Значение |
|------------|----------|
| mAP50      | **0.713** |
| mAP50-95   | **0.317** |
| Precision  | 0.744    |
| Recall     | 0.680    |
| Лучшая эпоха | 45 / 85 |
| Время      | 4 ч (Apple M4 Pro) |

---

## Шаг 1: Подготовка датасета

Скачать в формате **YOLO26** с Roboflow:

- **Smoking Person Detection** → распаковать в `Training/datasets/smoking_detection/`
  https://universe.roboflow.com/project-i6bzi/smoking-person-detection-ec7ec

---

## Шаг 2: Обучение

```bash
cd Training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Скачать базовые веса YOLO26n
# yolo26n.pt должен лежать в Training/

python train.py
# best.pt сохраняется в runs/detect/runs/cigarette_yolo26n/weights/best.pt
```

---

## Шаг 3: Экспорт в CoreML

```bash
python export_coreml.py
# создаёт cigarette_yolo26n.mlpackage в папке Training/
```

---

## Шаг 4: Проверка модели

```bash
python check_coreml_detections.py --image /path/to/test.jpg
# выводит детекции и сохраняет *_coreml_pred.png с нарисованными bbox
```

---

## Системные требования

| Компонент | Версия |
|-----------|--------|
| Python    | 3.9+   |
| PyTorch   | 2.0+   |
| Ultralytics | 8.4+ |
| macOS (MPS) / CUDA | — |
