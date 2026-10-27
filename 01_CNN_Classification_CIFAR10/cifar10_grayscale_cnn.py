import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert RGB images to grayscale
def rgb_to_grayscale(images):
    return np.dot(images[...,:3], [0.2989, 0.5870, 0.1140]).astype("float32")

x_train_gray = rgb_to_grayscale(x_train)
x_test_gray = rgb_to_grayscale(x_test)

def normalize(X_train, X_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = tf.image.per_image_standardization(X_train)
    X_test = tf.image.per_image_standardization(X_test)
    return X_train, X_test

x_train_gray, x_test_gray = normalize(x_train_gray, x_test_gray)

x_train_gray = np.expand_dims(x_train_gray, axis=-1)
x_test_gray = np.expand_dims(x_test_gray, axis=-1)

x_train_gray = x_train_gray[:15000]
y_train = y_train[:15000]

data_gen = ImageDataGenerator(
    horizontal_flip=True,     
    preprocessing_function=lambda x: x * 1.2  
)

def create_cnn_model():
    model = models.Sequential()
    # Layer L1 
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1), 
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Layer L2 
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Layer L3 
    model.add(layers.Conv2D(64, (3, 3), activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((2, 2)))
    
    # Fully Connected Layers 
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', 
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dense(10))
    
    return model

model = create_cnn_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    data_gen.flow(x_train_gray, y_train, batch_size=8),  
    epochs=100,
    validation_data=(x_test_gray, y_test)
)

test_loss, test_acc = model.evaluate(x_test_gray, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

y_pred = model.predict(x_test_gray)
y_pred_classes = np.argmax(y_pred, axis=1)  

f1 = f1_score(y_test, y_pred_classes, average='weighted')  
print(f"F1-score: {f1}")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
