import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CẤU HÌNH ---
MODEL_PATH = r"C:\Users\Nguyen Thien Khai\Downloads\Prj DL\cifar10_mobilenetv2_final.keras"
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def generate_report():
    """Tải model lên, đánh giá trên tập test, và in ra các báo cáo thống kê."""
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ LỖI: Không tìm thấy file model tại: {MODEL_PATH}")
        print("Hãy đảm bảo bạn đã chạy file train và file model nằm cùng thư mục.")
        return

    # 1. Tải Model & Dữ liệu Test
    print(">>> Đang tải Model và Dữ liệu Test...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Tải tập test gốc
    (_, _), (x_test_raw, y_test_original) = tf.keras.datasets.cifar10.load_data()
    
    # Chuẩn hóa dữ liệu test giống như lúc train (BẮT BUỘC)
    x_test = tf.keras.applications.mobilenet_v2.preprocess_input(x_test_raw.astype('float32'))
    y_test_true = y_test_original.flatten()
    y_test_cat = tf.keras.utils.to_categorical(y_test_original, 10)

    # 2. Đánh giá Model (Tính toán Loss và Accuracy cuối cùng)
    print(">>> Đang đánh giá hiệu suất cuối cùng...")
    loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print("\n--- KẾT QUẢ TỔNG HỢP ---")
    print(f"Final Test Loss: {loss:.4f} | Final Test Accuracy: {acc*100:.2f}%")
    
    # 3. Dự đoán toàn bộ tập Test (Để tạo bảng thống kê)
    print(">>> Đang dự đoán toàn bộ tập Test để tạo báo cáo chi tiết...")
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. In Báo cáo Phân loại (Classification Report)
    print("\n--- BÁO CÁO PHÂN LOẠI CHI TIẾT (PRECISION, RECALL, F1-SCORE) ---")
    print(classification_report(y_test_true, y_pred, target_names=CLASS_NAMES))

    # 5. Vẽ Ma trận Nhầm lẫn (Confusion Matrix)
    print(">>> Đang vẽ Ma trận Nhầm lẫn...")
    cm = confusion_matrix(y_test_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (True)')
    plt.title(f'Confusion Matrix (Final Accuracy: {acc*100:.2f}%)')
    plt.show()

if __name__ == "__main__":
    generate_report()