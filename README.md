# Spectacles.ai

An end-to-end Flask application for advanced image enhancement—deblurring and super-resolution—using classical computer vision and deep learning techniques.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Overview

Spectacles.ai processes user-uploaded images through two stages:

1. **Deblurring**: Applies an FFT-based low-pass filter to remove image blur, followed by CLAHE contrast enhancement.
2. **Super-Resolution**: Upscales the deblurred image using pretrained EDSR deep learning models (×2, ×3, ×4).

It also computes quality metrics (PSNR, SSIM, contrast gain, and processing time) and displays results via a user-friendly web interface.

## Features

- **Classical & AI Hybrid Pipeline**: Combines FFT-based deblurring with deep learning super-resolution.
- **Multiple Scales**: Supports ×2, ×3, and ×4 upscaling.
- **Quality Metrics**: Computes PSNR, SSIM, contrast gain, and processing time.
- **Web Interface**: Simple Flask frontend with Tailwind CSS and vanilla JavaScript.
- **Reporting**: View the last processed image and its metrics on `/report`.

## Tech Stack

- **Backend**: Python 3.x, Flask, OpenCV, scikit-image
- **Deep Learning**: OpenCV DNN Super-Resolution (EDSR models)
- **Frontend**: HTML, Tailwind CSS, JavaScript

## Project Structure

```bash
Spectacles.ai/
├── app.py               # Flask application and routes
├── deblurring.py        # Image processing utilities (deblur, gray conversion, upscaling)
├── models/              # Pretrained EDSR .pb model files (x2, x3, x4)
├── static/
│   └── uploads/         # Folder for saving user-uploaded and output images
├── templates/
│   ├── index.html       # Main upload and result page
│   └── report.html      # Last-upload report page
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/otAdarshP/Spectacles.ai.git
   cd Spectacles.ai
   ```
2. **Create & activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

- **Model files**: Pretrained EDSR models (`EDSR_x2.pb`, `EDSR_x3.pb`, `EDSR_x4.pb`) are included in `models/`.
- **Upload folder**: Defaults to `static/uploads/`. Change in `app.config['UPLOAD_FOLDER']` if needed.
- **Secret key**: Defined in `app.secret_key` for session management—update for production use.

## Running the Application

```bash
# Ensure your virtual environment is activated
python app.py
```

By default, Flask runs on `http://127.0.0.1:5000` in debug mode.

## Usage

1. **Upload an image** on the homepage.
2. Wait for processing—deblurred and upscaled images will display along with metrics.
3. **Download** processed images from the interface or check the uploads folder.
4. View the latest upload report at `http://127.0.0.1:5000/report`.

## API Endpoints

- `GET /` : Renders the upload form.
- `POST /` : Processes the uploaded image.
- `GET /report` : Shows last processed image and its metrics.

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -m 'Add my feature'`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

