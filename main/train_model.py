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
LEARNING_RATE = 0.0001
EPOCHS = 10

def build_data_augmentation_layers():
    """Defines a Keras Sequential model for on-the-fly data augmentation."""
    return Sequential([
        # Randomly flip images horizontally
        RandomFlip("horizontal"), 
        # Randomly rotate images by up to 20%
        RandomRotation(0.2),
        # Randomly zoom in/out by up to 20%
        RandomZoom(0.2), 
        # Rescale pixel values from [0, 255] to the [-1, 1] range expected by MobileNetV2
        # This is CRITICAL for using pre-trained weights
        Rescaling(1./127.5, offset=-1)
    ], name="data_augmentation")


def build_and_train_model(train_ds, val_ds, num_classes):
    """
    Builds the MobileNetV2 transfer learning model, compiles it, and starts training.
    Uses real Keras logic if TensorFlow is installed, otherwise prints mock messages.
    """
    if IS_REAL_TF:
        print("\n--- Building Real Keras Model with Data Augmentation ---")
        
        # 1. Define the augmentation pipeline
        data_augmentation = build_data_augmentation_layers()

        # 2. Load the pre-trained MobileNetV2 model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False, # We want to add our own classification layers
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        )
        # Freeze the base layers so they are not re-trained
        base_model.trainable = False 

        # 3. Construct the full model pipeline
        inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        # Augmentation is applied first
        x = data_augmentation(inputs)  
        # Then the frozen base model processes the augmented images
        x = base_model(x, training=False) 
        x = GlobalAveragePooling2D()(x) # Reduce feature maps to a single vector
        x = Dense(1024, activation='relu')(x) # Hidden dense layer
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # 4. Final Model
        model = Model(inputs=inputs, outputs=predictions)
        
        # 5. Compile the model
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"Real Keras Model compiled for {num_classes} classes.")
        
        # 6. Training
        print(f"\n--- Beginning Training (Real Keras) for {EPOCHS} epochs ---")
        try:
            model.fit(
                train_ds,
                epochs=EPOCHS,
                validation_data=val_ds
            )
            print("Training completed.")
        except Exception as e:
            print(f"Error during Keras model fit: {e}")
            print("This usually happens if the data pipeline or GPU setup is incomplete.")
            return

        # 7. Save the model
        model.save(MODEL_SAVE_PATH)
        print(f"\nModel saved successfully to {MODEL_SAVE_PATH}")

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
    train_ds, val_ds = load_and_prepare_data()

    if train_ds and hasattr(train_ds, 'class_names'):
        class_names = train_ds.class_names
        num_classes = len(class_names)
        
        print(f"\nFinal Class Count: {num_classes} classes")
        
        # 2. Build and Train
        build_and_train_model(train_ds, val_ds, num_classes)
    else:
        print("Error: Data loading failed or class names could not be determined. Aborting training.")


if __name__ == "__main__":
    # Ensure correct working directory context
    if os.path.basename(os.getcwd()) != 'main':
        print("Warning: Ensure you run this script from the 'main' directory.")
        
    main()
