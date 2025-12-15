import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, applications # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# --- CẤU HÌNH ---
IMG_SIZE = 96  # Resize ảnh lên 96x96 để MobileNet nhìn rõ
BATCH_SIZE = 64
EPOCHS_PHASE_1 = 10 # Giai đoạn Warm-up
EPOCHS_PHASE_2 = 15 # Giai đoạn Fine-tuning
MODEL_FILE_NAME = 'cifar10_mobilenetv2_final.keras' # Tên file model sẽ lưu

# 1. LOAD DATA
def load_data():
    print(">>> [1/4] Đang tải dữ liệu CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Chuẩn hóa theo chuẩn MobileNetV2
    x_train = applications.mobilenet_v2.preprocess_input(x_train.astype('float32'))
    x_test = applications.mobilenet_v2.preprocess_input(x_test.astype('float32'))

    y_train = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test_cat), y_test

# 2. XÂY DỰNG MODEL
def build_model():
    print(">>> [2/4] Đang xây dựng MobileNetV2...")
    inputs = tf.keras.Input(shape=(32, 32, 3)) # Input gốc 32x32
    
    # Phóng to ảnh & Augmentation
    x = layers.Resizing(IMG_SIZE, IMG_SIZE)(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomFlip("horizontal")(x)

    # Transfer Learning
    base_model = applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False # Khóa
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    return models.Model(inputs, outputs), base_model

# 3. CHƯƠNG TRÌNH CHÍNH
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test_cat), y_test_original = load_data()
    model, base_model = build_model()

    # --- TỰ ĐỘNG LƯU (QUAN TRỌNG NHẤT) ---
    # ModelCheckpoint: Sẽ lưu đè file .keras mỗi khi tìm thấy model ngon hơn
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_FILE_NAME, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # Phase 1: Train đầu
    print("\n>>> [3/4] TRAINING PHASE 1 (Frozen)...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    h1 = model.fit(x_train, y_train, epochs=EPOCHS_PHASE_1, batch_size=BATCH_SIZE, 
                   validation_data=(x_test, y_test_cat), callbacks=[checkpoint])

    # Phase 2: Fine-tuning
    print("\n>>> [4/4] TRAINING PHASE 2 (Fine-tuning)...")
    base_model.trainable = True
    for layer in base_model.layers[:-40]: layer.trainable = False # Chỉ mở 40 lớp cuối
    
    model.compile(optimizer=optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    h2 = model.fit(x_train, y_train, epochs=EPOCHS_PHASE_2, batch_size=BATCH_SIZE, 
                   validation_data=(x_test, y_test_cat), callbacks=[checkpoint, early_stop])

    print(f"\n✅ ĐÃ HOÀN TẤT! Model tốt nhất đã được lưu tại: {MODEL_FILE_NAME}")
    
    # Báo cáo kết quả
    print(">>> Đang tạo báo cáo...")
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = y_test_original.flatten()
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Vẽ biểu đồ (Vẽ sau cùng để ko chặn luồng lưu file)
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    plt.plot(acc, label='Train'); plt.plot(val_acc, label='Val')
    plt.title('Accuracy History'); plt.legend()
    plt.show()