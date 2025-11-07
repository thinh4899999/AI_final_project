import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')

print("STARTING FOOD RECOGNITION TRAINING WITH CUSTOM CNN")
drive.mount('/content/drive')
data_root = '/content/drive/MyDrive/AI/do_an'

# DATA COLLECTION
def collect_datasets():
    classes = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path) and any(img.lower().endswith(('.jpg', '.jpeg', '.png')) for img in os.listdir(item_path)):
            classes.append(item)
    return sorted(classes)

def collect_data(split):
    filepaths, labels = [], []
    for class_name in classes:
        class_dir = os.path.join(data_root, class_name, split)
        if os.path.exists(class_dir):
            for img in os.listdir(class_dir):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(class_dir, img))
                    labels.append(class_name)
    return pd.DataFrame({'filename': filepaths, 'class': labels})

classes = collect_datasets()
num_classes = len(classes)
df_train = collect_data('train')
df_val = collect_data('val')
df_test = collect_data('test')

print(f"📊 Dataset: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
print(f"🎯 Classes: {classes}")

# DATA GENERATOR
class FoodDataGenerator(Sequence):
    def __init__(self, dataframe, target_size=(224, 224), batch_size=32, shuffle=True, augment=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indices]
        
        X = np.zeros((len(batch_data), *self.target_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_data), num_classes), dtype=np.float32)
        
        for i, (_, row) in enumerate(batch_data.iterrows()):
            img = self.load_and_preprocess_image(row['filename'])
            if self.augment:
                img = self.augment_image(img)
            X[i] = img
            y[i, self.class_to_idx[row['class']]] = 1.0
        return X, y

    def load_and_preprocess_image(self, filepath):
        img = cv2.imread(filepath)
        if img is None:
            return np.zeros((*self.target_size, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        img = img / 255.0  # Normalize to [0, 1]
        return img.astype(np.float32)

    def augment_image(self, img):
        # Random flip
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Random brightness and contrast
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-0.1, 0.1)  # Brightness
            img = cv2.convertScaleAbs(img * 255, alpha=alpha, beta=beta) / 255.0
        
        return img

    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)

# DATA PREPARATION
target_size = (224, 224)
batch_size = 32

# Clean data
df_train = df_train[df_train['filename'].apply(os.path.exists)]
df_val = df_val[df_val['filename'].apply(os.path.exists)]
df_test = df_test[df_test['filename'].apply(os.path.exists)]

train_gen = FoodDataGenerator(df_train, target_size, batch_size, shuffle=True, augment=True)
val_gen = FoodDataGenerator(df_val, target_size, batch_size, shuffle=False)
test_gen = FoodDataGenerator(df_test, target_size, batch_size, shuffle=False)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(df_train['class']), y=df_train['class'])
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

# CUSTOM CNN MODEL ARCHITECTURE
def create_custom_cnn():
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*target_size, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fifth Convolutional Block
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Fully Connected Layers
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# CREATE AND COMPILE MODEL
model = create_custom_cnn()

# Display model architecture
model.summary()

# MODEL COMPILATION
model.compile(
    optimizer=AdamW(learning_rate=0.001, weight_decay=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# 📈 TRAINING SETUP
callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_custom_cnn_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    CSVLogger('custom_cnn_training_log.csv')
]

# TRAINING EXECUTION
print("🚀 TRAINING CUSTOM CNN MODEL")
history = model.fit(
    train_gen,
    epochs=100,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# EVALUATION
print("=== FINAL EVALUATION ===")
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_gen, verbose=0)
print(f"✅ TEST ACCURACY: {test_accuracy:.4f}")
print(f"✅ TEST PRECISION: {test_precision:.4f}")
print(f"✅ TEST RECALL: {test_recall:.4f}")

# Predictions
y_true, y_pred = [], []
for i in range(len(test_gen)):
    X_batch, y_batch = test_gen[i]
    preds = model.predict(X_batch, verbose=0)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=classes, digits=3))

# Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title(f'Custom CNN Confusion Matrix - Accuracy: {test_accuracy:.4f}')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_custom_cnn.png', dpi=300, bbox_inches='tight')
plt.show()

# Training history visualization
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history_custom_cnn.png', dpi=300, bbox_inches='tight')
plt.show()

# SAVE MODEL
model.save('CUSTOM_CNN_FOOD_RECOGNITION_MODEL.keras')
print("✅ CUSTOM CNN MODEL SAVED SUCCESSFULLY!")
print("🎉 TRAINING COMPLETED!")

# Display model architecture info
print(f"\n📐 MODEL ARCHITECTURE INFO:")
print(f"   - Input shape: {target_size} x 3")
print(f"   - Number of classes: {num_classes}")
print(f"   - Total parameters: {model.count_params():,}")
print(f"   - Number of layers: {len(model.layers)}")
