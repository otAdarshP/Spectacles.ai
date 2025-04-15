import os
import numpy as np
import time
import cv2
from flask import Flask, render_template, request
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from deblurring import load_image, convert_to_gray, deblur_fft, upscale_image
from flask import session
  # required to use session


app = Flask(__name__)

app.secret_key = 'Mahi@7781'

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/report")
def report():
    if 'last_upload' not in session:
        return render_template("report.html", error="No recent upload to report.")
    return render_template("report.html", data=session['last_upload'])


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")

        if file is None or file.filename == "":
            return render_template("index.html", error="No file selected. Please upload a valid image.")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            start_time = time.time()

            # Stage 1: Load and deblur
            img = load_image(filepath)
            gray = convert_to_gray(img)
            deblurred = deblur_fft(gray)

            # Stage 2: Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            deblurred = clahe.apply(deblurred)

            # Optional: Invert bright backgrounds
            if np.mean(deblurred) > 200:
                deblurred = cv2.bitwise_not(deblurred)

            # Convert grayscale to BGR
            deblurred_bgr = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)

            # Save deblurred image
            deblurred_path = os.path.join(app.config['UPLOAD_FOLDER'], "deblurred_" + file.filename)
            cv2.imwrite(deblurred_path, deblurred)

            # Super-resolution models
            models = {
                "x2": ("models/EDSR_x2.pb", "edsr", 2),
                "x3": ("models/EDSR_x3.pb", "edsr", 3),
                "x4": ("models/EDSR_x4.pb", "edsr", 4)
            }

            enhanced_paths = {}
            psnr_score = None
            ssim_score = None

            for label, (path, name, scale) in models.items():
                enhanced = upscale_image(deblurred_bgr, path, name, scale)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"enhanced_{label}_" + file.filename)
                cv2.imwrite(save_path, enhanced)
                enhanced_paths[label] = f"enhanced_{label}_" + file.filename

                if label == "x3":
                     enhanced_gray = convert_to_gray(enhanced)
                     # Resize gray to match enhanced_gray
                     if gray.shape != enhanced_gray.shape:
                          resized_gray = cv2.resize(gray, (enhanced_gray.shape[1], enhanced_gray.shape[0]))
                     else:
                          resized_gray = gray
                     psnr_score = psnr(resized_gray, enhanced_gray)
                     ssim_score = ssim(resized_gray, enhanced_gray)


            # Calculate contrast gain
            before_contrast = np.std(gray)
            after_contrast = np.std(deblurred)
            contrast_gain = after_contrast - before_contrast

            # Total processing time
            processing_time = round(time.time() - start_time, 3)

            session['last_upload'] = {
                "original": file.filename,
                "deblurred": "deblurred_" + file.filename,
                "enhanced_x3": enhanced_paths.get("x3"),
                "metrics": {
                    "psnr": round(psnr_score, 2) if psnr_score else None,
                    "ssim": round(ssim_score, 3) if ssim_score else None,
                    "time": processing_time,
                    "contrast_gain": round(contrast_gain, 2)
                }
            }


            return render_template("index.html",
                                   original=file.filename,
                                   deblurred="deblurred_" + file.filename,
                                   enhanced_paths=enhanced_paths,
                                   metrics={
                                       "psnr": round(psnr_score, 2) if psnr_score else None,
                                       "ssim": round(ssim_score, 3) if ssim_score else None,
                                       "time": processing_time,
                                       "contrast_gain": round(contrast_gain, 2)
                                   },
                                   psnr_val=round(psnr_score, 2) if psnr_score else None,
                                   ssim_val=round(ssim_score, 3) if ssim_score else None)

        except Exception as e:
            return render_template("index.html", error=f"Processing failed: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
