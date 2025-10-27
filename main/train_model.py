import os
import sys
# Import the centralized configuration and dependency handling from data_prep
from data_prep import load_and_prepare_data, SEED, tf

# Check if we are running the real TensorFlow environment
IS_REAL_TF = hasattr(tf, 'version')

if IS_REAL_TF:
    # Only import necessary Keras components if the real TF is available
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, Rescaling
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.optimizers import Adam
    # Callbacks (EarlyStopping, ModelCheckpoint) removed
    # Set seed early for real TensorFlow
    tf.random.set_seed(SEED)
    print("TensorFlow initialization complete.")
else:
    # If not real TF, inform the user we are using mock utilities
    print("\nNOTE: Proceeding with mock training. Install TensorFlow and related dependencies for real training.")
    # The mock tf object already handles set_seed when imported from data_prep.

# --- Model Configuration ---
MODEL_SAVE_PATH = 'trashnet_classifier.keras'
IMAGE_SIZE = (224, 224)

# Learning Rates
LEARNING_RATE = 0.0001        # Rate for training the new head layers

# Training Phases
EPOCHS = 10         # Train the new head layer only (reverting to simple training)

def build_data_augmentation_layers():
    """Defines a Keras Sequential model for on-the-fly data augmentation."""
    return Sequential([
        # Randomly flip images horizontally
        RandomFlip("horizontal"), 
        # Randomly rotate images up to 20%
        RandomRotation(0.2), 
        # Randomly zoom images up to 20%
        RandomZoom(0.2)
    ])

def build_model(num_classes):
    """
    Builds the MobileNetV2 classification model with custom top layers.
    The base model remains frozen, enabling stable Transfer Learning.
    """
    
    # 1. Load the pre-trained MobileNetV2 base model once
    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers (This is the defining state for the model)
    base_model.trainable = False 

    # Create the full model using Sequential API
    model = Sequential([
        # Rescale inputs to the [0, 1] range (MobileNetV2 expects this)
        Rescaling(1./255),
        
        # Add data augmentation layers for on-the-fly processing
        build_data_augmentation_layers(),
        
        # Add the frozen base model
        base_model,
        
        # New classification head:
        # Global Average Pooling to reduce spatial dimensions
        GlobalAveragePooling2D(),
        
        # ADDED: A small, dense hidden layer to increase model capacity
        Dense(128, activation='relu'), 
        
        # Output layer with softmax activation for multi-class classification
        Dense(num_classes, activation='softmax')
    ])

    return model

def build_and_train_model(train_ds, val_ds, num_classes):
    """Compiles and trains the Keras model, only training the new classification head."""

    if IS_REAL_TF:

        # 1. Build the model with a frozen base
        model = build_model(num_classes)
        
        print("\n--- Starting Model Training (Head Layers Only) ---")
        print(f"Total Epochs: {EPOCHS}")

        # 2. Compile the model
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n--- Model Summary ---")
        model.summary()

        # 3. Train
        model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds
        )
        
        # 4. Save the model
        model.save(MODEL_SAVE_PATH)
        print(f"\nModel saved successfully to {MODEL_SAVE_PATH} after {EPOCHS} epochs.")

    else:
        # Mock training execution (prints status messages only)
        print(f"\n--- Building Mock Model for {num_classes} classes ---")
        print("Mock: Model Summary (TF not installed)")
        print("Mock: Model Compile")
        print("\n--- Beginning Training ---")
        print(f"Mock: Model Fit - Training for {EPOCHS} epochs...")
        print("Mock: Training completed. Mock model saved to disk.")


def main():
    print("\n--- Starting Model Training Process ---")
    
    # 1. Load data (this function also handles download/cleanup if necessary)
    result = load_and_prepare_data()

    if result and len(result) == 3:
        train_ds, val_ds, class_names = result
        
        # Check if the data objects themselves are valid (not None from failure)
        if train_ds is None or val_ds is None:
            print("Error: Dataset objects are None, indicating a FATAL ERROR during Keras setup. Aborting training.")
            return

        num_classes = len(class_names)
        
        print(f"\nFinal Class Count: {num_classes} classes")
        
        # 2. Build and Train
        build_and_train_model(train_ds, val_ds, num_classes)
        
    else:
        # Generic error handler for data loading issues
        print("Error: Data loading failed or returned an invalid structure. Aborting training.")


if __name__ == "__main__":
    main()
