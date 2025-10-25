### SmartSort
SmartSort: Deep Learning Trash ClassifierThis project implements a deep learning model (based on MobileNetV2) to classify waste images into six categories (Cardboard, Glass, Metal, Paper, Plastic, Trash). The workflow includes data preparation, model training, and an interactive test harness for local and remote prediction testing.PrerequisitesEnsure you have Python 3.8+ installed.1. Environment SetupIt is highly recommended to use a virtual environment:# Create the environment
python -m venv venv

## Activate the environment (Linux/macOS)
source venv/bin/activate

## Activate the environment (Windows)
.\venv\Scripts\activate

2. Install Dependencies
You will need TensorFlow, NumPy, and PIL (Pillow) for image handling, plus requests for the remote test function.
pip install tensorflow numpy Pillow requests

Project StructureThis guide assumes the following directory structure:SmartSort/
├── main/
│   ├── data_prep.py        # Data download and loading
│   ├── train_model.py      # Model definition and training
│   ├── handler.py          # Core prediction logic (handles base64 input)
│   ├── log_config.py       # Centralized logger configuration
│   └── trashnet_classifier.keras (Generated after training)
└── test/
    ├── TestApp.py          # Interactive local/remote testing harness
    └── metal28.jpg         # Example image for testing

## Step 1: Data Preparation
The data_prep.py script automatically downloads, extracts, and cleans the necessary TrashNet dataset.
Command:python main/data_prep.py
Output:This creates the trashnet_data/dataset-resized directory containing the images.

## Step 2: Model Training
The train_model.py script loads the data, builds the MobileNetV2-based model with data augmentation, and trains it for 10 epochs.
Command:python main/train_model.py
Output:This generates the final trained model file in the main/ directory: trashnet_classifier.keras.
## Step 3: Interactive Testing 
Use the TestApp.py harness to interactively test your trained model locally or check connectivity to a remote OpenFaaS endpoint.

# A. Set Environment Variables (Optional, but Recommended)
To see detailed logs (which are saved to a file) and enable diagnostics during the local test, set these variables:VariableValuePurposeDIAGNOSTICStrueEnables detailed logging.LOG_FILESmartSort.logSets the log file name (saved in /tmp).
Example (Linux/macOS):export DIAGNOSTICS=true 
export LOG_FILE=SmartSort.log
Example (Windows - Command Prompt):set DIAGNOSTICS=true
set LOG_FILE=SmartSort.log

# B. Run the Test HarnessCommand:python test/TestApp.py
The application will prompt you for a choice:
Enter l (Local): Executes the predict.py's handle function directly, loading the trashnet_classifier.keras file from the main/ directory and classifying the image.
Enter r (Remote): Attempts to send the test image to the remote OpenFaaS endpoint (http://localhost:8080/function/ocr-detect).
Enter e (Exit): Closes the application.C. Check LogsIf DIAGNOSTICS is set to true, a log file will be generated:

# View the logs after running a local test
cat /tmp/SmartSort.log

