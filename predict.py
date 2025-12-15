import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Tên file model (phải trùng với tên trong file train)
MODEL_PATH = r"C:\Users\Nguyen Thien Khai\Downloads\Prj DL\cifar10_mobilenetv2_final.keras"
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def predict_external_image(image_path):
    """Hàm dự đoán một file ảnh bất kỳ từ máy tính"""
    if not os.path.exists(MODEL_PATH):
        print("❌ LỖI: Không tìm thấy file model (.keras). Hãy chạy train_cifar10.py trước!")
        return

    # 1. Tải Model (Chỉ mất 1 giây)
    print(f"Loading model từ {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không đọc được ảnh tại: {image_path}")
        return

    # 3. Xử lý ảnh để hiển thị
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 4. Xử lý ảnh để đưa vào model (QUAN TRỌNG)
    # Resize về kích thước tương đối (MobileNet có lớp resize bên trong nhưng nên resize trước để nhẹ)
    img_input = cv2.resize(img_rgb, (32, 32)) 
    # Preprocess chuẩn MobileNetV2
    img_input = tf.keras.applications.mobilenet_v2.preprocess_input(img_input.astype('float32'))
    img_input = np.expand_dims(img_input, axis=0) # Thêm chiều batch -> (1, 32, 32, 3)

    # 5. Dự đoán
    preds = model.predict(img_input)
    score = np.max(preds)
    label = CLASS_NAMES[np.argmax(preds)]

    # 6. Hiển thị kết quả
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.title(f"AI Dự đoán: {label}\nĐộ tin cậy: {score*100:.2f}%", color='green', fontsize=14)
    plt.axis('off')
    plt.show()

# --- KHU VỰC CHẠY THỬ ---

# CÁCH 1: Dự đoán ảnh tải từ mạng về (Bạn bỏ comment dòng dưới và thay tên file ảnh vào)
predict_external_image(r"C:\Users\Nguyen Thien Khai\Downloads\Prj DL\con_tau.jpg")

# CÁCH 2: Test nhanh bằng dữ liệu có sẵn (Nếu chưa có ảnh ngoài)
print("\n>>> Đang test thử với 1 ảnh ngẫu nhiên từ bộ Test Set...")
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
idx = np.random.randint(0, len(x_test))
sample_img = x_test[idx] # Lấy ảnh gốc
true_label = CLASS_NAMES[y_test[idx][0]]

# Lưu tạm thành file ảnh để giả lập việc load từ bên ngoài
cv2.imwrite("temp_test_image.jpg", cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR))
predict_external_image("temp_test_image.jpg")
print(f"(Nhãn thực tế của ảnh này là: {true_label})")