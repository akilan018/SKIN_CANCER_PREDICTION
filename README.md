# Skin Cancer Prediction - AI/ML Application

A deep learning web application that classifies skin lesion images into 7 different categories using EfficientNetB3 trained on the HAM10000 dataset.

## 🎯 Features

- **AI-Powered Detection**: Uses transfer learning with EfficientNetB3
- **7 Classification Types**:
  - Actinic Keratosis
  - Basal Cell Carcinoma
  - Benign Keratosis
  - Dermatofibroma
  - Melanoma
  - Melanocytic Nevus (Normal Skin)
  - Vascular Lesion
- **Web Interface**: User-friendly frontend for image upload and analysis
- **Real-time Predictions**: Instant classification with confidence scores

## 📋 Prerequisites

- **Python**: 3.9 or higher (Python 3.10+ recommended)
- **Disk Space**: At least 500 MB free
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Internet**: Required for initial package installation

## 🚀 Installation & Setup

### Step 1: Extract the Project
```bash
# Unzip the project to your desired location
# Navigate to the project folder
cd path/to/SKIN_CANCER_PREDICTION
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes depending on your internet speed as TensorFlow is ~400 MB.

## ▶️ Running the Application

### Start the Server
```bash
python api.py
```

You should see output like:
```
Loading best_model.h5 on TF 2.17.0
✓ Model loaded successfully!
 * Running on http://127.0.0.1:5000
```

### Access the Application
1. Open your web browser
2. Navigate to: **http://127.0.0.1:5000**
3. Upload a skin lesion image
4. Click "Analyze Image"
5. View the prediction results

### Stop the Server
Press `Ctrl + C` in the terminal where the server is running.

## 📁 Project Structure

```
SKIN_CANCER_PREDICTION/
├── api.py                 # Flask backend server
├── best_model.h5          # Pre-trained AI model (46 MB)
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── style.css         # UI styles
│   └── script.js         # Frontend logic
└── README.md             # This file
```

## 🔧 Troubleshooting

### Model Loading Fails
**Error**: "Model not loaded on the server"

**Solutions**:
1. Ensure `best_model.h5` is in the same directory as `api.py`
2. Check if TensorFlow 2.17.0 is installed: `pip show tensorflow-cpu`
3. Reinstall TensorFlow: `pip install --upgrade tensorflow-cpu==2.17.0`

### Port Already in Use
**Error**: "Address already in use"

**Solutions**:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

### Import Errors
**Error**: "ModuleNotFoundError: No module named 'flask'"

**Solution**:
```bash
pip install -r requirements.txt --force-reinstall
```

### Python Version Issues
If you're using Python 3.9, you may see warnings. The app will still work, but upgrading to Python 3.10+ is recommended:
- Download from: https://www.python.org/downloads/

## 🖥️ System Requirements

### Minimum
- CPU: Dual-core processor
- RAM: 4 GB
- Python: 3.9
- OS: Windows 7+, Ubuntu 18.04+, macOS 10.14+

### Recommended
- CPU: Quad-core processor
- RAM: 8 GB
- Python: 3.10+
- OS: Windows 10+, Ubuntu 20.04+, macOS 11+

## 📦 Dependencies

Main packages installed via `requirements.txt`:
- **Flask 3.1.3**: Web framework
- **TensorFlow-CPU 2.17.0**: Deep learning library
- **Keras 3.10.0**: Neural network API
- **Pillow 10.2.0**: Image processing
- **NumPy 1.26.4**: Numerical computing
- **Flask-CORS 6.0.2**: Cross-origin requests
- **Gunicorn 21.2.0**: Production server (optional)

## 🎓 Model Information

- **Architecture**: Sequential with EfficientNetB3 base
- **Total Parameters**: 10,981,174
- **Trainable Parameters**: 197,639
- **Input Size**: 224×224×3 (RGB images)
- **Output**: 7-class classification
- **Framework**: Keras 3.8.0 with TensorFlow backend
- **Training Dataset**: HAM10000

## ⚠️ Disclaimer

**For Educational Use Only**

This application is designed for educational and demonstration purposes. It should **NOT** be used for actual medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical concerns.

## 🚀 Production Deployment (Optional)

For production use, replace the development server with Gunicorn:

```bash
# Already included in requirements.txt
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

For cloud deployment (e.g., Render, Railway, Heroku), the included `render.yaml` provides configuration.

## 📞 Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Ensure all prerequisites are met
3. Verify Python version: `python --version`
4. Check installed packages: `pip list`

## 📄 License

This project is provided for educational purposes.

---

**Built with ❤️ using Flask, TensorFlow, and EfficientNetB3**
