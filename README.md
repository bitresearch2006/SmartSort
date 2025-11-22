# SmartSort - Biomedical Waste Classification System

**SmartSort** is a deep learning-based biomedical waste classification system that uses MobileNetV2 to automatically categorize medical waste images into 11 different categories. The system provides both local and remote inference capabilities with comprehensive model evaluation tools.

## 🏥 Categories Supported

- **(BT) Body Tissue or Organ** - Biological tissues and organ waste
- **(GE) Glass equipment-packaging 551** - Glass laboratory equipment and packaging
- **(ME) Metal equipment-packaging** - Metal medical equipment and packaging
- **(OW) Organic wastes** - Organic biological waste materials
- **(PE) Plastic equipment-packaging** - Plastic medical equipment and packaging
- **(PP) Paper equipment-packaging** - Paper-based medical packaging
- **Gauze** - Medical gauze and similar fabric materials
- **Gloves** - Medical gloves (latex, nitrile, etc.)
- **Mask** - Medical masks and face coverings
- **Syringe** - Medical syringes and needles
- **Tweezers** - Medical tweezers and similar instruments

## 📁 Project Structure

```
SmartSort/
├── main/
│   ├── categories.json         # Category definitions and metadata
│   ├── config.json            # Model and training configuration
│   ├── config_loader.py       # Configuration management utilities
│   ├── data_prep.py           # Data preparation and preprocessing
│   ├── train_model.py         # Model training script
│   ├── handler.py             # Prediction handler for local/remote inference
│   ├── log_config.py          # Logging configuration
│   └── requirements.txt       # Python dependencies
├── test/
│   └── TestApp.py             # Interactive testing application
├── data/
│   └── biomedical_dataset/    # Raw training data
└── SmartSort.yml              # OpenFaaS deployment configuration
```

## 🛠️ Step 1: Install Dependencies

### Prerequisites
- **Python 3.8+**
- **Git** (for cloning the repository)

### 1.1 Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv smartsort-env

# Activate environment
# Windows:
smartsort-env\Scripts\activate
# Linux/macOS:
source smartsort-env/bin/activate
```

### 1.2 Install Required Packages

```bash
# Navigate to project directory
cd SmartSort/main

# Install all dependencies
pip install -r requirements.txt
```

**Dependencies include:**
- `tensorflow` - Deep learning framework
- `numpy` - Numerical computing
- `pillow` - Image processing
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation
- `opencv-python` - Computer vision (for camera capture)
- `requests` - HTTP client for remote testing

## 📊 Step 2: Data Preparation

### 2.1 Organize Your Data

Place your biomedical waste images in the following structure:
```
data/biomedical_dataset/
├── (BT) Body Tissue or Organ/
├── (GE) Glass equipment-packaging 551/
├── (ME) Metal equipment-packaging/
├── (OW) Organic wastes/
├── (PE) Plastic equipment-packaging/
├── (PP) Paper equipment-packaging/
├── Gauze/
├── Gloves/
├── Mask/
├── Syringe/
└── Tweezers/
```

### 2.2 Run Data Preparation

```bash
# Navigate to main directory
cd main

# Run data preparation
python data_prep.py
```

**What happens:**
- ✅ Scans the raw dataset directory
- ✅ Automatically detects categories from folder names
- ✅ Updates `categories.json` with found categories
- ✅ Flattens nested directory structure for Keras compatibility
- ✅ Creates train/validation split (80/20)
- ✅ Generates preprocessed dataset in `../data/biomedical_dataset_FLAT/`

**Output:**
```
--- Starting Data Preparation for Biomedical Dataset ---
--- Flattening Dataset Structure for Keras compatibility ---
Processing category: Gauze
Processing category: Gloves
Processing category: Mask
...
--- Flattening Complete. Total files moved: 5991 ---
Found 5335 files belonging to 11 classes.
```

## 🚀 Step 3: Model Training

### 3.1 Start Training

```bash
# Make sure you're in the main directory
cd main

# Start training
python train_model.py
```

### 3.2 Training Process

**Architecture:**
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Input Size:** 224x224x3 pixels
- **Data Augmentation:** Random flip, rotation, zoom
- **Optimizer:** Adam (learning rate: 0.0001)
- **Loss Function:** Categorical crossentropy
- **Training Epochs:** 10 (with early stopping)

**What happens during training:**
1. ✅ Loads preprocessed data (4,268 training + 1,067 validation images)
2. ✅ Builds MobileNetV2-based model with transfer learning
3. ✅ Applies data augmentation for better generalization
4. ✅ Trains with early stopping and model checkpointing
5. ✅ Automatically saves best model as `biomedical_waste_classifier.keras`
6. ✅ Generates comprehensive evaluation metrics

**Expected Output:**
```
--- Starting Model Training Process ---
Final Class Count: 11 classes
--- Beginning Training for 10 epochs ---
Epoch 1/10: loss: 0.8234 - accuracy: 0.7543 - val_loss: 0.3456 - val_accuracy: 0.8901
...
Training completed. The best model was saved to biomedical_waste_classifier.keras
Model metrics generated successfully. Check the ./metrics directory for results.
```

### 3.3 Training Results

After training, you'll get:
- **Model file:** `biomedical_waste_classifier.keras`
- **Metrics directory** with evaluation results:
  - Confusion matrix visualization
  - Classification report (text + JSON)
  - Per-class performance metrics

## 🧪 Step 4: Testing with Different Modes

The TestApp provides multiple ways to test your trained model:

### 4.1 Interactive Mode (Recommended)

```bash
# Navigate to test directory
cd test

# Run interactive mode
python TestApp.py
```

**Interactive Menu Options:**
```
📋 Options:
1. Load image file          # Load existing image file
2. Capture from camera      # Capture new image from camera
3. List available cameras   # Check available cameras
4. Show current image       # Display loaded image
5. Test local              # Test with local model
6. Test remote             # Test with remote OpenFaaS endpoint
7. Exit                    # Exit application
```

### 4.2 Command Line Mode

#### Test with Image File (Local)
```bash
# Test local model with image file
python TestApp.py --image mask.jpg --mode local

# Test remote endpoint with image file
python TestApp.py --image syringe.jpg --mode remote
```

#### Test with Camera Capture
```bash
# Capture from default camera and test locally
python TestApp.py --camera --mode local

# Use specific camera (e.g., camera index 1)
python TestApp.py --camera --camera-index 1 --mode local

# Capture and test remotely
python TestApp.py --camera --mode remote
```

#### List Available Cameras
```bash
# Check available cameras in your system
python TestApp.py --list-cameras
```

### 4.3 Testing Modes Explained

#### **Local Mode**
- ✅ Uses the trained model directly (`biomedical_waste_classifier.keras`)
- ✅ Faster inference (no network latency)
- ✅ Works offline
- ✅ Full debugging capabilities

#### **Remote Mode**
- ✅ Tests OpenFaaS deployment
- ✅ Simulates production environment
- ✅ Network-based inference
- ⚠️ Requires OpenFaaS deployment (see deployment section)

### 4.4 Example Test Output

```bash
🚀 SmartSort - Command Line Mode
==================================================
✅ Image loaded successfully: mask.jpg
📊 Base64 size: 353040 characters

[Image Display Window Opens]

🖥️  Testing with LOCAL model...
----------------------------------------
🎯 LOCAL CLASSIFICATION RESULT:
==================================================
📋 Classification: Mask
🎯 Confidence: 99.68%

📊 Probability Breakdown:
------------------------------
Mask                               ████████████████████  99.7%
Gauze                              ░░░░░░░░░░░░░░░░░░░░   0.1%
Gloves                             ░░░░░░░░░░░░░░░░░░░░   0.1%
...
```

### 4.5 Camera Capture Instructions

When using camera mode:
1. **Position your item** clearly in front of the camera
2. **Ensure good lighting** for better classification
3. **Press SPACE** to capture the image
4. **Press ESC** to cancel capture
5. The captured image will be **automatically displayed**
6. Classification will **run immediately** after capture

## 🔧 Troubleshooting

### Common Issues:

**1. "No module named 'tensorflow'"**
```bash
pip install tensorflow
```

**2. "Camera not available"**
```bash
# List available cameras
python TestApp.py --list-cameras

# Try different camera index
python TestApp.py --camera --camera-index 1
```

**3. "Model file not found"**
```bash
# Make sure you've trained the model first
cd main
python train_model.py
```

**4. "Could not connect to remote endpoint"**
- Remote mode requires OpenFaaS deployment
- For local testing, use `--mode local`

**5. Low classification confidence**
- Ensure good image quality and lighting
- Check if the item is clearly visible
- Verify the item belongs to one of the 11 supported categories

## 📈 Model Performance

**Achieved Results:**
- **Overall Accuracy:** 99%
- **Validation Samples:** 1,067 images
- **Training Samples:** 4,268 images
- **Perfect Classes:** Metal equipment, Mask, Syringe, Tweezers (100% F1-score)

## 🚀 Deployment (Optional)

For production deployment using OpenFaaS, see `SmartSort.yml` configuration file.

## 📝 Configuration Files

- **`categories.json`**: Category definitions and metadata
- **`config.json`**: Model and training parameters
- **`requirements.txt`**: Python dependencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**🏥 SmartSort - Making biomedical waste classification intelligent and efficient!**

