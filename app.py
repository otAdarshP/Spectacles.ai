import os
import time
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, session
from werkzeug.utils import secure_filename
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from cv2 import dnn_superres

app = Flask(__name__)
app.secret_key = 'spectacles_secret_key'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load EDSR model once
ds_model = dnn_superres.DnnSuperResImpl_create()
ds_model.readModel("models/EDSR_x3.pb")
ds_model.setModel("edsr", 3)

def enhance_image(img_path):
    original = cv2.imread(img_path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Deblurring via Wiener Filter (basic implementation)
    psf = np.ones((5, 5)) / 25
    psf = psf / psf.sum()
    blurred_fft = np.fft.fft2(gray)
    psf_fft = np.fft.fft2(psf, s=gray.shape)
    restored = np.abs(np.fft.ifft2(blurred_fft / (psf_fft + 1e-3)))
    restored = np.uint8(np.clip(restored, 0, 255))

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(restored)

    # Super-resolution
    sr = ds_model.upsample(original)

    # Save outputs
    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)
    deblurred_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_deblurred{ext}")
    enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_enhanced{ext}")
    cv2.imwrite(deblurred_path, enhanced_gray)
    cv2.imwrite(enhanced_path, sr)

    # Metrics
    psnr_val = psnr(gray, enhanced_gray)
    ssim_val = ssim(gray, enhanced_gray)
    contrast_gain = float(np.std(enhanced_gray) - np.std(gray))

    return deblurred_path, enhanced_path, psnr_val, ssim_val, contrast_gain

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start = time.time()
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            deblurred_path, enhanced_path, psnr_val, ssim_val, contrast_gain = enhance_image(filepath)
            elapsed = round(time.time() - start, 3)

            session['results'] = {
                'original': filename,
                'deblurred': os.path.basename(deblurred_path),
                'enhanced': os.path.basename(enhanced_path),
                'psnr': round(psnr_val, 2),
                'ssim': round(ssim_val, 3),
                'contrast_gain': round(contrast_gain, 2),
                'time': elapsed
            }

            return redirect(url_for('results'))

    return render_template('index.html')

@app.route('/results')
def results():
    if 'results' not in session:
        return redirect(url_for('index'))
    return render_template('results.html', metrics=session['results'])

@app.route('/report')
def report():
    return render_template('report.html', metrics=session.get('results', {}))

if __name__ == '__main__':
    app.run(debug=True)
