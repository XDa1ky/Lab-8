import cv2
import numpy as np
import os

def overlay_image_alpha(background, overlay, x, y):
    # Размеры фона и накладываемого изображения
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]

    # Если часть накладываемого изображения выходит за границы фона, подрезаем
    if x >= bg_w or y >= bg_h:
        return background  # ничего не накладываем, т.к. за пределами
    if x + ol_w > bg_w:
        ol_w = bg_w - x
        overlay = overlay[:, :ol_w]
    if y + ol_h > bg_h:
        ol_h = bg_h - y
        overlay = overlay[:ol_h]

    # Разделяем каналы RGBA
    if overlay.shape[2] == 4:
        ol_b, ol_g, ol_r, ol_a = cv2.split(overlay)
        alpha = ol_a.astype(float) / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])
        ol_rgb = cv2.merge([ol_b, ol_g, ol_r])
    else:
        # Если нет альфа-канала, просто копируем поверх
        ol_rgb = overlay
        alpha = np.ones((ol_h, ol_w, 3), dtype=float)

    roi = background[y:y+ol_h, x:x+ol_w].astype(float)

    blended = alpha * ol_rgb.astype(float) + (1.0 - alpha) * roi
    background[y:y+ol_h, x:x+ol_w] = blended.astype(np.uint8)

    return background

def main():
    # Пути к файлам
    template_path = "ref-point.jpg"    
    fly_path      = "fly64.png"        

    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Шаблон метки не найден по пути: {template_path}")
    if not os.path.isfile(fly_path):
        raise FileNotFoundError(f"Изображение мухи не найдено по пути: {fly_path}")

    # Загружаем шаблон (градационное изображение)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    t_h, t_w = template.shape[:2]

    # Загружаем муху (сохраняем альфа-канал)
    fly = cv2.imread(fly_path, cv2.IMREAD_UNCHANGED)
    if fly is None or fly.shape[2] != 4:
        raise ValueError("fly64.png должен содержать 4 канала (RGBA).")

    f_h, f_w = fly.shape[:2]

    # Запускаем видеопоток с камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру.")

    # Порог для детекции метки
    threshold = 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразуем кадр в оттенки серого для корреляции
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            # Нашли метку: вычисляем её центр
            top_left = max_loc
            center_x = top_left[0] + t_w // 2
            center_y = top_left[1] + t_h // 2

            # Координаты для наложения мухи: так, чтобы её центр совпал с центром метки
            x_offset = center_x - f_w // 2
            y_offset = center_y - f_h // 2

            # Накладываем муху на кадр
            frame = overlay_image_alpha(frame, fly, x_offset, y_offset)

            # (Опционально) можно нарисовать контуры найденной метки
            bottom_right = (top_left[0] + t_w, top_left[1] + t_h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # Отображаем результат
        cv2.imshow("Frame with Fly Overlay", frame)

        # Выход по нажатию 'w'
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
