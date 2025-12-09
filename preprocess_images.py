import cv2
import numpy as np


def preprocess_digit(image_bytes):
    # convert raw bytes to image (grayscale)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None, None

    # blur image to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # make image black digit on white background (like MNIST)
    _, thresh = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # find all contours (shapes)
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None

    # use the biggest contour (assume it is the digit)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # cut out just the digit region
    digit = thresh[y:y + h, x:x + w]

    # resize digit to around 20x20 while keeping shape ratio
    h_old, w_old = digit.shape

    if h_old > w_old:
        scale = 20 / h_old
        new_h = 20
        new_w = int(w_old * scale)
    else:
        scale = 20 / w_old
        new_w = 20
        new_h = int(h_old * scale)

    digit = cv2.resize(digit, (new_w, new_h))

    # pad image to exactly 28x28
    pad_h = 28 - new_h
    pad_w = 28 - new_w

    padded = np.pad(
        digit,
        (
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2)
        ),
        mode="constant",
        constant_values=0
    )

    # scale pixels to 0–1
    norm = padded.astype("float32") / 255.0

    # reshape for CNN: (batch_size, height, width, channels)
    norm = norm.reshape(1, 28, 28, 1)

    return norm, padded
