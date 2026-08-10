import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Create models folder automatically
os.makedirs("models", exist_ok=True)


# Dataset Paths
train_dir = "Facemaskdetection/train"
val_dir = "Facemaskdetection/val"


IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 10


# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)


val_datagen = ImageDataGenerator(
    rescale=1.0/255
)


# Load Dataset
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)


val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)


# IMPORTANT: Check Label Mapping
print("\nClass Mapping:")
print(train_data.class_indices)



# CNN Model
model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(128,128,3)
    ),

    MaxPooling2D(2,2),


    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),


    Conv2D(
        128,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),


    Flatten(),


    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),


    Dense(
        1,
        activation="sigmoid"
    )

])



model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)



model.summary()



# Stop if model stops improving
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)



# Training
history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    callbacks=[early_stop]

)



# Save Model
model.save(
    "models/mask_detector.keras"
)



print("\n===================================")
print("Training Completed Successfully!")
print("Model Saved: models/mask_detector.keras")
print("===================================")



# Plot Result

plt.figure(figsize=(10,4))


plt.subplot(1,2,1)

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Accuracy")
plt.legend()



plt.subplot(1,2,2)

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Loss")
plt.legend()



plt.tight_layout()

plt.savefig(
    "models/training_result.png"
)

plt.show()