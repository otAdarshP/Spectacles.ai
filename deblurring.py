# import cv2
# import numpy as np
# from cv2 import dnn_superres

# def load_image(image_path):
#     """Load image from the given path."""
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Unable to load image from {image_path}")
#     return img

# def convert_to_gray(img):
#     """Convert image to grayscale."""
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# def deblur_fft(img_gray, radius=30):
#     """Deblur image using FFT low-pass filtering."""
#     # Step 1: Fourier Transform
#     dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
#     dft_shift = np.fft.fftshift(dft)

#     # Step 2: Create low-pass circular mask
#     rows, cols = img_gray.shape
#     crow, ccol = rows // 2, cols // 2
#     mask = np.ones((rows, cols), np.uint8)
#     x, y = np.ogrid[:rows, :cols]
#     mask_area = (x - crow)**2 + (y - ccol)**2 <= radius**2
#     mask[mask_area] = 0

#     # Step 3: Apply mask and inverse FFT
#     mask = np.expand_dims(mask, axis=-1)
#     mask = np.repeat(mask, 2, axis=-1)
#     fshift = dft_shift * mask
#     f_ishift = np.fft.ifftshift(fshift)
#     img_back = cv2.idft(f_ishift)

#     # Step 4: Convert complex result to magnitude
#     deblurred = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

#     # Step 5: Normalize to 0–255 and convert to uint8
#     deblurred = cv2.normalize(deblurred, None, 0, 255, cv2.NORM_MINMAX)
#     return deblurred.astype(np.uint8)

# def upscale_image(img, scale=3, model_path='models/EDSR_x3.pb', model_name='edsr'):
#     """Upscale image using a pre-trained super-resolution model."""
#     sr = dnn_superres.DnnSuperResImpl_create()
#     sr.readModel(model_path)
#     sr.setModel(model_name, scale)
#     return sr.upsample(img)

# deblurring.py

import cv2
import numpy as np
from cv2 import dnn_superres

def load_image(image_path):
    """Load image from the given path."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to load image from {image_path}")
    return img

def convert_to_gray(img):
    """Convert image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def deblur_fft(img_gray, radius=30):
    """Deblur image using FFT low-pass filtering."""
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= radius**2
    mask[mask_area] = 0

    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 2, axis=-1)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    deblurred = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    deblurred = cv2.normalize(deblurred, None, 0, 255, cv2.NORM_MINMAX)
    return deblurred.astype(np.uint8)

def upscale_image(img, model_path, model_name, scale):
    """Upscale image using a pre-trained super-resolution model."""
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model_name, scale)
    return sr.upsample(img)
