import os
from flask import Flask, render_template, request
import cv2
from deblurring import load_image, convert_to_gray, deblur_fft, upscale_image

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded file safely
        file = request.files.get("image")

        # Validate file
        if file is None or file.filename == "":
            return render_template("index.html", error="No file selected. Please upload a valid image.")

        # Save uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Stage 1: Deblurring
            img = load_image(filepath)
            gray = convert_to_gray(img)
            deblurred = deblur_fft(gray)

            # Convert to BGR before super-resolution
            deblurred_bgr = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)

            # Stage 2: Super-resolution
            enhanced = upscale_image(deblurred_bgr)

            # Save processed images
            deblurred_path = os.path.join(app.config['UPLOAD_FOLDER'], "deblurred_" + file.filename)
            final_path = os.path.join(app.config['UPLOAD_FOLDER'], "enhanced_" + file.filename)

            cv2.imwrite(deblurred_path, deblurred)
            cv2.imwrite(final_path, enhanced)

            return render_template("index.html",
                                   original=file.filename,
                                   deblurred="deblurred_" + file.filename,
                                   enhanced="enhanced_" + file.filename)
        except Exception as e:
            return render_template("index.html", error=f"Processing failed: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)



# import os
# from flask import Flask, render_template, request
# import cv2
# from deblurring import load_image, convert_to_gray, deblur_fft, upscale_image

# app = Flask(__name__)

# # Configure upload folder
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         # Get uploaded file safely
#         file = request.files.get("image")

#         # Validate file
#         if file is None or file.filename == "":
#             return render_template("index.html", error="No file selected. Please upload a valid image.")

#         # Save uploaded image
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)

#         try:
#             # Stage 1: Deblurring
#             img = load_image(filepath)
#             gray = convert_to_gray(img)
#             deblurred = deblur_fft(gray)

#             # Convert to BGR before super-resolution
#             deblurred_bgr = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)

#             # Stage 2: Super-resolution
#             enhanced = upscale_image(deblurred_bgr)
#             enhances_loop = upscale_image(deblurred)

#             # Save processed images
#             deblurred_path = os.path.join(app.config['UPLOAD_FOLDER'], "deblurred_" + file.filename)
#             final_path = os.path.join(app.config['UPLOAD_FOLDER'], "enhanced_" + file.filename)

#             cv2.imwrite(deblurred_path, deblurred)
#             cv2.imwrite(final_path, enhanced)

#             return render_template("index.html",
#                                    original=file.filename,
#                                    deblurred="deblurred_" + file.filename,
#                                    enhanced="enhanced_" + file.filename)
#         except Exception as e:
#             return render_template("index.html", error=f"Processing failed: {str(e)}")

#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)

