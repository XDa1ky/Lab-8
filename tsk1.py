import cv2
import numpy as np

def add_gaussian_noise(img, sigma=30):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # Укажите свой путь и расширение:
    path_in  = r"images\variant-5.jpg"
    path_out = r"images\variant-5_noisy.jpg"

    img = cv2.imread(path_in)
    if img is None:
        raise FileNotFoundError(f"Не могу открыть {path_in}")

    noisy = add_gaussian_noise(img, sigma=30)
    cv2.imwrite(path_out, noisy)

    # Опционально: показать результат
    cv2.imshow("Noisy", noisy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
