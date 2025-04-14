# Spectacles.ai
 Becoming the spectacles of the images (or the image viewer, too).

What It Does:

User uploads an image via the browser.

Backend: Reads the image → Converts to grayscale → Deblurs using FFT. Converts grayscale back to BGR → Applies Super-Resolution using pre-trained deep learning models (cv2.dnn_superres).

Outputs: Original image --> Deblurred image --> Enhanced high-res image

Features:

💻 Frontend: Easy-to-use UI using Flask and HTML.

🧠 Backend: Modular, reusable, clean Python functions.

🚀 AI-Enhanced: Uses Super-Resolution for real improvement.

🌐 Ready for deployment (Render/Heroku).

📂 Downloads possible — save outputs in static/uploads.
