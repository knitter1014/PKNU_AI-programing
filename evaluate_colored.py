import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import os
import numpy as np

# 데이터 로드 및 전처리
def load_images_and_labels_from_directory(directory, target_size=(150, 150)):
    x_data = []
    y_data = []
    labels = []
    
    for label_name in sorted(os.listdir(directory)):
        label_path = os.path.join(directory, label_name)
        if not os.path.isdir(label_path):
            continue
        
        labels.append(label_name)
        label_index = len(labels) - 1
        
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                img = tf.keras.preprocessing.image.load_img(file_path, target_size=target_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                x_data.append(img_array)
                y_data.append(label_index)
    
    return np.array(x_data), np.array(y_data), labels

def load_data(testing_dir, target_size=(150, 150)):
    x_test, y_test, labels = load_images_and_labels_from_directory(testing_dir, target_size)
    num_classes = len(labels)
    
    y_test = to_categorical(y_test, num_classes)
    
    return x_test, y_test, num_classes

# 경로 설정
testing_dir = "./testing(colored)"

# 데이터 로드
x_test, y_test, num_classes = load_data(testing_dir)

# 저장된 모델 로드
model = load_model('./model_colored/densenet_custom5.h5')
print("저장된 모델을 로드했습니다.")

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n테스트 정확도: {test_acc:.4f}')