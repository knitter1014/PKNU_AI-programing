import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json

# 데이터 로드 및 전처리
def load_labels_from_json(label_file):
    with open(label_file, 'r') as f:
        data = json.load(f)
    
    image_paths = []
    labels = []
    for file_info in data['files']:
        image_paths.append(file_info['path'])
        labels.append(file_info['label']['label'])
    
    return image_paths, labels

def load_images_from_paths(image_paths, labels, target_size=(150, 150)):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)  # 레이블을 정수로 변환
    
    images = []
    for img_path in image_paths:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # 이미지를 정규화
        images.append(img_array)
    
    return np.array(images), np.array(encoded_labels), label_encoder.classes_

def load_data(testing_dir, label_file, target_size=(150, 150)):
    # 레이블 로드
    image_paths, labels = load_labels_from_json(label_file)
    
    # 테스트 데이터 분리
    test_image_paths = [path for path in image_paths if path.startswith('testing')]
    test_labels = [labels[i] for i, path in enumerate(image_paths) if path.startswith('testing')]
    
    # 이미지 로드 및 레이블 인코딩
    x_test, y_test, classes = load_images_from_paths(test_image_paths, test_labels, target_size)
    
    y_test = to_categorical(y_test, len(classes))
    
    return x_test, y_test, len(classes)

# 경로 설정
testing_dir = "./testing"
label_file = "./info.labels"

# 데이터 로드
x_test, y_test, num_classes = load_data(testing_dir, label_file)

# 저장된 모델 로드
model = load_model('./model/densenet_custom6.h5')
print("저장된 모델을 로드했습니다.")

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n테스트 정확도: {test_acc:.4f}')