import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
image_path = "image.jpg"

if not os.path.exists(image_path):
    print("Error: image.jpg not found!")
    print("Place your image.jpg in the same folder as this Python file.")
    exit()

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Unable to read the image.")
    exit()

print("Image loaded successfully.")
print("Image size:", img.shape)

dft = cv2.dft(
    np.float32(img),
    flags=cv2.DFT_COMPLEX_OUTPUT
)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = cv2.magnitude(
    dft_shift[:, :, 0],
    dft_shift[:, :, 1]
)

magnitude_spectrum = 20 * np.log(
    magnitude_spectrum + 1
)
rows, cols = img.shape

crow = rows // 2
ccol = cols // 2
radius = 50

low_pass_mask = np.zeros((rows, cols, 2), np.float32)

cv2.circle(
    low_pass_mask,
    (ccol, crow),
    radius,
    (1, 1),
    -1
)


low_pass_dft = dft_shift * low_pass_mask

low_pass_ishift = np.fft.ifftshift(low_pass_dft)

low_pass_image = cv2.idft(low_pass_ishift)

low_pass_image = cv2.magnitude(
    low_pass_image[:, :, 0],
    low_pass_image[:, :, 1]
)

# Normalize image
low_pass_image = cv2.normalize(
    low_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

low_pass_image = np.uint8(low_pass_image)
high_pass_mask = np.ones((rows, cols, 2), np.float32)

cv2.circle(
    high_pass_mask,
    (ccol, crow),
    radius,
    (0, 0),
    -1
)


high_pass_dft = dft_shift * high_pass_mask

high_pass_ishift = np.fft.ifftshift(high_pass_dft)

high_pass_image = cv2.idft(high_pass_ishift)

high_pass_image = cv2.magnitude(
    high_pass_image[:, :, 0],
    high_pass_image[:, :, 1]
)

high_pass_image = cv2.normalize(
    high_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

high_pass_image = np.uint8(high_pass_image)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Frequency Spectrum")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(low_pass_image, cmap="gray")
plt.title("Low-Pass Filtered Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(high_pass_image, cmap="gray")
plt.title("High-Pass Filtered Image")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("low_pass_result.jpg", low_pass_image)
cv2.imwrite("high_pass_result.jpg", high_pass_image)
cv2.imwrite("frequency_spectrum.jpg",
            np.uint8(cv2.normalize(
                magnitude_spectrum,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )))

print("Results saved successfully!")