import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ------------------------
# 1. Data Loading
# ------------------------
def load_data(base_dir, img_size=(128, 128)):
    images = []
    labels = []
    class_names = sorted(os.listdir(base_dir))  # Class names from folder names
    class_to_index = {name: idx for idx, name in enumerate(class_names)}  # Map class to index

    for class_name in class_names:
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    img_path = os.path.join(class_path, img_file)
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size, color_mode='grayscale')
                    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0, 1]
                    images.append(img)
                    labels.append(class_to_index[class_name])

    return np.array(images), np.array(labels), class_names

# Path to your extracted dataset
base_dir = r"C:\Users\job01\Desktop\lung_new_project\archive(32)\archive(32)\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset\train"

# Load dataset
img_size = (128, 128)
images, labels, class_names = load_data(base_dir, img_size)

# One-hot encode labels
n_class = len(class_names)
labels = to_categorical(labels, num_classes=n_class)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# ------------------------
# 2. Define the Capsule Network
# ------------------------
def CapsNet(input_shape, n_class):
    x_input = layers.Input(shape=input_shape)

    # Convolutional Layer
    x = layers.Conv2D(256, kernel_size=9, strides=1, padding='valid', activation='relu')(x_input)

    # Primary Capsule Layer
    x = layers.Conv2D(128, kernel_size=9, strides=2, padding='valid', activation='relu')(x)
    x = layers.Reshape((-1, 8))(x)  # Reshape into capsules: (batch_size, num_capsules, capsule_dim)

    # Flatten capsules
    x = layers.Flatten()(x)

    # Fully Connected Capsule Layer
    caps1 = layers.Dense(128, activation='relu')(x)

    # Digit Capsule Layer
    caps2 = layers.Dense(n_class, activation='softmax')(caps1)

    # Model
    model = models.Model(inputs=x_input, outputs=caps2)
    return model

# Create the Capsule Network model
input_shape = (img_size[0], img_size[1], 1)  # Grayscale image shape
model = CapsNet(input_shape=input_shape, n_class=n_class)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# ------------------------
# 3. Train the Model
# ------------------------
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

# ------------------------
# 4. Print Training and Validation Accuracy
# ------------------------
# Print the training and validation accuracy
print("Training and Validation Accuracy over Epochs:")
for epoch in range(len(history.history['accuracy'])):
    print(f"Epoch {epoch+1}: Train Accuracy = {history.history['accuracy'][epoch]:.4f}, "
          f"Validation Accuracy = {history.history['val_accuracy'][epoch]:.4f}")

# ------------------------
# 5. Save the Model
# ------------------------
# Save the trained model as an HDF5 file
model.save('capsnet_lung_cancer_model.h5')
