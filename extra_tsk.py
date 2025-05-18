import cv2
import numpy as np

# Загрузка шаблона метки
template = cv2.imread('ref-point.jpg', cv2.IMREAD_GRAYSCALE)
if template is None:
    print("Ошибка: не найден файл шаблона!")
    exit()

w, h = template.shape[::-1]

# Инициализация видеопотока с камеры
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Поиск шаблона
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    threshold = 0.7
    if max_val >= threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center_x = top_left[0] + w // 2
        center_y = top_left[1] + h // 2

        # Определяем цвет рамки по положению центра
        if center_x <= 50 and center_y <= 50:
            color = (255, 0, 0)  # Синий
        elif center_x >= frame_width - 50 and center_y >= frame_height - 50:
            color = (0, 0, 255)  # Красный
        else:
            color = (0, 255, 0)  # Зелёный (по умолчанию)

        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
        cv2.putText(frame, f"Center: ({center_x}, {center_y})", (center_x + 10, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Tracking with Variant 5', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
