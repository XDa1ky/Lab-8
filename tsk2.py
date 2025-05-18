import cv2
import numpy as np

# Загрузка шаблона метки
template = cv2.imread('ref-point.jpg', cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

# Захват видео с камеры
cap = cv2.VideoCapture(0)  # 0 — это вебкамера

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Поиск шаблона на кадре
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Порог достоверности
    threshold = 0.7
    if max_val >= threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Центр метки
        center = (top_left[0] + w // 2, top_left[1] + h // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Center: {center}", (center[0]+10, center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
