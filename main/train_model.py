import os
import sys
import numpy as np
# Use modular configuration instead of data_prep dependency
from config_loader import (
    load_categories, get_model_config, get_training_config, 
    get_data_config, get_image_size, get_num_classes
)
from data_prep import load_and_prepare_data  # Only for data loading functionality

# Load configuration from JSON files
CATEGORIES = load_categories()
MODEL_CONFIG = get_model_config()
TRAINING_CONFIG = get_training_config()
DATA_CONFIG = get_data_config()

# Extract configuration values
IMAGE_SIZE = get_image_size()
SEED = MODEL_CONFIG.get('seed', 42)
LEARNING_RATE = MODEL_CONFIG.get('learning_rate', 0.0001)
EPOCHS = MODEL_CONFIG.get('epochs', 10)
BATCH_SIZE = MODEL_CONFIG.get('batch_size', 32)

# Try to import TensorFlow and check availability
try:
    import tensorflow as tf
    IS_REAL_TF = hasattr(tf, 'version')
    if IS_REAL_TF:
        tf.random.set_seed(SEED)
        print("TensorFlow initialization complete.")
except ImportError:
    tf = None
    IS_REAL_TF = False
    print("\nNOTE: TensorFlow not available. Proceeding with mock training.")

if IS_REAL_TF:
    # Only import necessary Keras components if the real TF is available
    try:
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, Rescaling
        from tensorflow.keras.models import Model, Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        print("Keras components imported successfully.")
    except ImportError as e:
        print(f"Error importing Keras components: {e}")
        IS_REAL_TF = False

# --- Model Configuration ---
# Updated model name to reflect the dataset
MODEL_SAVE_PATH = MODEL_CONFIG.get('name', 'biomedical_waste_classifier.keras')

def build_data_augmentation_layers():
    """Defines a Keras Sequential model for on-the-fly data augmentation."""
    return Sequential([
        # Randomly flip images horizontally
        RandomFlip("horizontal"), 
        # Randomly rotate images by up to 20%
        RandomRotation(0.2),
        # Randomly zoom in or out on images by up to 20%
        RandomZoom(0.2),
    ], name="data_augmentation_layers")

def build_model(num_classes):
    """
    Builds the MobileNetV2 transfer learning model with data augmentation.
    """
    # 1. Load the pre-trained MobileNetV2 base model
    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), 
        include_top=False, 
        weights='imagenet'
    )

    # 2. Freeze the base model layers 
    base_model.trainable = False

    # 3. Create the model structure
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    
    # Preprocessing layer: rescale pixel values from [0, 255] to [-1, 1] as required by MobileNetV2
    x = Rescaling(1./127.5, offset=-1)(inputs) 
    
    # Add Data Augmentation
    x = build_data_augmentation_layers()(x)
    
    # Pass through the MobileNetV2 base model
    x = base_model(x, training=False)
    
    # Add new classification layers (the 'head')
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(128, activation='relu')(x) 
    
    # Final output layer
    outputs = Dense(num_classes, activation='softmax')(x) 

    model = Model(inputs, outputs)
    return model


def build_and_train_model(train_ds, val_ds, num_classes):
    """
    Compiles, trains, and saves the classification model.
    """
    if IS_REAL_TF:
        # 1. Build the model
        model = build_model(num_classes)
        
        # 2. Compile the model
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print summary
        print("\n--- Model Summary ---")
        model.summary()
        print("---------------------")

        # 3. Define Callbacks
        callbacks = [
            # Stop training early if validation loss hasn't improved for 3 epochs
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), 
            # Save the best model based on validation accuracy
            ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1) 
        ]
        
        # 4. Train the model
        print(f"\n--- Beginning Training for {EPOCHS} epochs ---")
        try:
            # We use the train_ds and val_ds directly from load_and_prepare_data
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                callbacks=callbacks
            )
            # The ModelCheckpoint callback handles saving the best version.
            print(f"\nTraining completed. The best model was saved to {MODEL_SAVE_PATH}")
            
            # 5. Generate comprehensive model metrics
            print("\n--- Generating Model Metrics ---")
            try:
                from model_metrics import generate_model_metrics
                # Load the best saved model for evaluation
                best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
                generate_model_metrics(best_model, val_ds, save_dir="./metrics")
                print("Model metrics generated successfully. Check the ./metrics directory for results.")
            except ImportError:
                print("Warning: Could not import model_metrics. Skipping metrics generation.")
            except Exception as e:
                print(f"Warning: Failed to generate metrics: {e}")
            
            return history
            
        except tf.errors.InvalidArgumentError as e:
            print(f"\nTensorFlow Error during training (InvalidArgumentError): {e}")
            print("This often indicates a mismatch in data pipeline, batch size, or GPU setup.")
            return

    else:
        # Mock training execution (prints status messages only)
        print(f"\n--- Building Mock Model for {num_classes} classes ---")
        print("Mock: Model Summary (TF not installed)")
        print("Mock: Model Compile")
        print("\n--- Beginning Training ---")
        print(f"Mock: Model Fit - Training for {EPOCHS} epochs...")
        
        # Mock model saving (since we use ModelCheckpoint callback now)
        with open(MODEL_SAVE_PATH, 'w') as f:
            f.write("Mock Model Data")
        print(f"Mock: Training completed. Mock model saved to disk at {MODEL_SAVE_PATH}")
        
        # Generate mock metrics
        print("\n--- Generating Mock Model Metrics ---")
        try:
            from model_metrics import generate_model_metrics
            generate_model_metrics(model=None, validation_dataset=None, save_dir="./metrics")
            print("Mock model metrics generated successfully. Check the ./metrics directory for results.")
        except ImportError:
            print("Warning: Could not import model_metrics. Skipping mock metrics generation.")
        except Exception as e:
            print(f"Warning: Failed to generate mock metrics: {e}")


def main():
    print("\n--- Starting Model Training Process ---")
    
    # 1. Load data (this function also handles download/cleanup if necessary)
    # train_ds and val_ds will either be real Keras datasets or MockDataset objects
    train_ds, val_ds = load_and_prepare_data()

    # Get the number of classes from the loaded categories
    num_classes = get_num_classes()
    
    # Check if the datasets were returned successfully AND we have classes
    # If the datasets are not None, training proceeds.
    if train_ds is not None and val_ds is not None and num_classes > 0:
        
        print(f"\nFinal Class Count: {num_classes} classes")
        
        # 2. Build and Train
        build_and_train_model(train_ds, val_ds, num_classes)
    else:
        # This error should now only trigger if load_and_prepare_data truly failed (e.g., raw data directory missing)
        print("Error: Data loading failed or returned an invalid structure. Aborting training.")


if __name__ == "__main__":
    main()