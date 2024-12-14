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

def load_data(training_dir, testing_dir, image_size=(150, 150)):
    def load_images_from_directory(directory):
        x_data = []
        y_data = []
        labels = []  # 디렉토리 이름 기반으로 라벨 추출
        
        for label_name in sorted(os.listdir(directory)):  # 디렉토리 이름 정렬
            label_path = os.path.join(directory, label_name)
            if not os.path.isdir(label_path):  # 디렉토리가 아니면 건너뜀
                continue
            
            # 디렉토리 이름을 라벨로 저장
            labels.append(label_name)
            label_index = len(labels) - 1  # 현재 라벨의 인덱스
            
            # 해당 디렉토리의 모든 이미지 로드
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if file_name.endswith(('.jpg', '.png', '.jpeg')):  # 이미지 파일만 처리
                    image = tf.keras.preprocessing.image.load_img(file_path, target_size=image_size)  # 이미지 로드 및 크기 조정
                    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # 정규화
                    x_data.append(image_array)
                    y_data.append(label_index)
        
        return np.array(x_data), np.array(y_data), labels

    # 학습 및 테스트 데이터 로드
    x_train, y_train, train_labels = load_images_from_directory(training_dir)
    print(train_labels)
    x_test, y_test, test_labels = load_images_from_directory(testing_dir)
    print(test_labels)
    
    # 라벨 매핑 검증
    assert train_labels == test_labels, "Training과 Testing 라벨이 일치하지 않습니다."
    
    # One-hot 인코딩
    num_classes = len(train_labels)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    return (x_train, y_train), (x_test, y_test), num_classes

# 경로 설정
training_dir = "../training(colored)"
testing_dir = "../testing(colored)"

# 데이터 로드
(x_train, y_train), (x_test, y_test), num_classes = load_data(training_dir, testing_dir)

# 모델 생성 및 컴파일
input_shape = (150, 150, 3)
model = create_densenet(input_shape=input_shape, num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping 콜백 추가
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # 검증 손실을 모니터링
    patience=5,  # 5번의 에폭 동안 개선되지 않으면 학습 종료
    restore_best_weights=True  # 최상의 모델 가중치를 복원
)

# 모델 학습 및 저장
start_time = time.time()
model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping])  # EarlyStopping 추가
end_time = time.time()

# 학습 시간 출력
print(f"모델 학습 시간: {end_time - start_time:.2f}초")

# 모델 저장
model.save('densenet_custom6.h5')
print("모델이 저장 완료")