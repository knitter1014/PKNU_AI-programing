import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import time

# Dense Block 정의
def dense_block(x, num_layers, growth_rate):
    for i in range(num_layers):
        bn = BatchNormalization()(x)
        act = Activation('relu')(bn)
        conv = Conv2D(growth_rate, kernel_size=(3, 3), padding='same')(act)
        x = Concatenate()([x, conv])
    return x

# Transition Layer 정의
def transition_layer(x, reduction):
    bn = BatchNormalization()(x)
    act = Activation('relu')(bn)
    num_filters = int(x.shape[-1] * reduction)
    conv = Conv2D(num_filters, kernel_size=(1, 1), padding='same')(act)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
    return x

# DenseNet 모델 정의
def create_densenet(input_shape, num_classes, growth_rate=12, num_dense_blocks=4, num_layers_per_block=6, reduction=0.5):
    inputs = Input(shape=input_shape)
    x = Conv2D(2 * growth_rate, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    for i in range(num_dense_blocks):
        x = dense_block(x, num_layers=num_layers_per_block, growth_rate=growth_rate)
        if i != num_dense_blocks - 1:
            x = transition_layer(x, reduction=reduction)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

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

# 이미지 로드 및 전처리
def load_images_from_paths(image_paths, labels, target_size=(150, 150)):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)  # 레이블을 정수로 변환
    
    images = []
    for img_path in image_paths:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # 이미지를 정규화
        images.append(img_array)
    
    return np.array(images), np.array(encoded_labels), label_encoder.classes_

def load_data(training_dir, testing_dir, label_file, target_size=(150, 150)):
    # 레이블 로드
    image_paths, labels = load_labels_from_json(label_file)
    
    # 훈련 데이터와 테스트 데이터 분리
    train_image_paths = [path for path in image_paths if path.startswith('training')]
    test_image_paths = [path for path in image_paths if path.startswith('testing')]
    
    train_labels = [labels[i] for i, path in enumerate(image_paths) if path.startswith('training')]
    test_labels = [labels[i] for i, path in enumerate(image_paths) if path.startswith('testing')]
    
    # 이미지 로드 및 레이블 인코딩
    x_train, y_train, classes = load_images_from_paths(train_image_paths, train_labels, target_size)
    x_test, y_test, _ = load_images_from_paths(test_image_paths, test_labels, target_size)
    
    y_train = to_categorical(y_train, len(classes))
    y_test = to_categorical(y_test, len(classes))
    
    return (x_train, y_train), (x_test, y_test), len(classes)

# 경로 설정
training_dir = "./training"
testing_dir = "./testing"
label_file = "./info.labels"

# 데이터 로드
(x_train, y_train), (x_test, y_test), num_classes = load_data(training_dir, testing_dir, label_file)

# 모델 생성 및 컴파일
input_shape = (150, 150, 3)
model = create_densenet(input_shape=input_shape, num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 및 저장
start_time = time.time()
model.fit(x_train, y_train, epochs=40, batch_size=32, validation_data=(x_test, y_test))  # 에폭을 20으로 증가
end_time = time.time()

# 학습 시간 출력
print(f"모델 학습 시간: {end_time - start_time:.2f}초")

# 모델 저장
model.save('densenet_custom.h5')
print("모델이 저장 완료")