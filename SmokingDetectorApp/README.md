# SmokingDetectorApp

iOS приложение для обнаружения и отслеживания фактов курения человеком по кадрам видеопотока с камеры в реальном масштабе времени.

Вариант задания: **28 — Обнаружение и отслеживание фактов курения человеком**

## Архитектура

- **YOLO26n** обучен на датасете [Smoking Person Detection](https://universe.roboflow.com/project-i6bzi/smoking-person-detection-ec7ec)
- Два класса: `person` (0) и `smoke` (1)
- Курение детектируется по пересечению bbox `smoke` с bbox `person`
- Модель экспортирована в **CoreML** (`cigarette_yolo26n.mlpackage`)

## Структура проекта

```
SmokingDetectorApp/
├── project.yml                          # спецификация XcodeGen
├── SmokingDetectorApp.xcodeproj/        # сгенерированный Xcode-проект
├── Sources/SmokingDetectorApp/
│   ├── AppDelegate.swift                # точка входа
│   ├── SceneDelegate.swift              # настройка окна
│   ├── ViewController.swift             # основной UI + координация
│   ├── CameraManager.swift              # AVFoundation камера
│   ├── SmokingDetector.swift            # CoreML/Vision инференс
│   ├── Detection.swift                  # модель детекции
│   ├── BoundingBoxOverlayView.swift     # отрисовка bbox
│   ├── LaunchScreen.storyboard
│   ├── Info.plist
│   └── Resources/
│       └── cigarette_yolo26n.mlpackage  # CoreML модель
```

## Системные требования

- iOS 16.0+
- Xcode 15.0+
- Устройство с камерой (iPhone / iPad)
- macOS для сборки в Xcode

## Сборка и запуск

1. Открыть `SmokingDetectorApp.xcodeproj` в Xcode
2. Подключить iPhone по USB (или использовать симулятор с фото)
3. Выбрать таргет-устройство и нажать **Run** (⌘+R)
4. При первом запуске на устройстве может потребоваться доверие в Settings → General → Device Management

## Ключевые алгоритмы

### 1. Инференс CoreML
- `VNCoreMLRequest` обрабатывает `CVPixelBuffer` с камеры
- Модель выдаёт тензор `[1, 300, 6]` (box + conf + cls)
- Ручной NMS: сортировка по confidence + IoU-фильтрация (`iouThreshold = 0.5`)

### 2. Логика курения
```swift
func checkSmoking(detections: [Detection]) -> Bool {
    let persons = detections.filter { $0.cls == .person }
    let smokes  = detections.filter { $0.cls == .smoke }
    for p in persons {
        for s in smokes where p.bbox.intersects(s.bbox) {
            return true
        }
    }
    return false
}
```

### 3. Оптимизация real-time
- Пропуск кадров: инференс не чаще чем каждые ~80 мс (12–13 FPS)
- `MLModelConfiguration.computeUnits = .all` (Neural Engine + GPU + CPU)
- `AVCaptureSessionPreset.vga640x480` для баланса скорости/качества

## Функциональные возможности

- [x] Потоковая обработка видео с камеры (передняя / задняя — переключение по кнопке)
- [x] Детекция людей и дыма/сигарет в реальном времени
- [x] Визуализация bounding boxes с confidence-скором
- [x] Индикация "SMOKING DETECTED!" при пересечении bbox
- [x] Отображение FPS и текущей камеры в углу экрана

## Что выводится при детекции

### Bounding Boxes
- **Person** — синий прямоугольник с confidence (например, `Person 0.85`)
- **Smoke** — оранжевый прямоугольник с confidence (например, `Smoke 0.72`)

### Индикация курения
Когда bbox класса `smoke` пересекается с bbox класса `person`:
- **Красный текст "SMOKING DETECTED!"** по центру верхней части экрана
- Чёрный полупрозрачный фон под текстом для читаемости

### Информационная панель
- **Левый верхний угол** — название приложения + текущая камера (Front/Back) + модель
- **Правый верхний угол** — текущий FPS (например, `12.3 FPS`)

### Переключение камеры
- **Круглая кнопка** внизу экрана с иконкой `camera.rotate.fill`
- Нажатие переключает между фронтальной и основной камерой
- Изображение с фронтальной камеры отзеркалено (mirror) для естественного отображения
