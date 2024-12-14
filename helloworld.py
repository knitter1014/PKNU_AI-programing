import cv2
import os
import numpy as np

folder_header = r'training'

# 이미지가 있는 원본 폴더 경로
input_folder = folder_header + r'(infrared)/'

# 처리된 이미지를 저장할 상위 폴더 경로
output_folder = folder_header + r'(infrared_processed_images)'

percent1to20 = r'1~20%'
percent21to40 = r'21~40%'
percent41to60 = r'41~60%'
percent61to80 = r'61~80%'
percent81to100 = r'81~100%'
percentList = [percent1to20, percent21to40, percent41to60, percent61to80, percent81to100]

# 상위 폴더가 없다면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 폴더 내 모든 이미지 파일 읽기
for percent in percentList:
    image_files = [f for f in os.listdir(input_folder + percent) if f.endswith('.jpg') or f.endswith('.png')]

    section_folder = os.path.join(output_folder, percent)
    label_file_path = os.path.join(section_folder, 'label.txt')  # 라벨 파일 경로
        
    # 하위 폴더가 없다면 생성
    if not os.path.exists(section_folder):
        os.makedirs(section_folder)

    # 현재 구간에 해당하는 파일들 처리
    for filename in image_files:
        input_image_path = os.path.join(input_folder + percent, filename)

        # 이미지를 읽어들임
        image = cv2.imread(input_image_path)
        
        if image is None:
            print(f"Error loading image: {filename}")
            continue
        
        # 빨간색 채널 강조하여 적외선 효과 적용
        red_channel = image[:, :, 2]  # BGR에서 빨간색 채널은 2번 인덱스
        image[:, :, 1] = 0  # 녹색 채널을 0으로 설정
        image[:, :, 0] = 0  # 파란색 채널을 0으로 설정
        
        # 그레이스케일로 변환하여 흑백 적외선 이미지 생성
        infrared_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 새로운 파일 경로 (구간별 폴더에 이미지 이름 유지)
        output_image_path = os.path.join(section_folder, filename)
        
        # 변환된 이미지를 저장
        cv2.imwrite(output_image_path, infrared_image)
        
        print(f"Processed {filename} (Section {percent}) and saved to {output_image_path}")

        # 라벨 정보를 텍스트 파일로 저장
        with open(label_file_path, 'w') as label_file:
            label_file.write(percent)
        print(f"Label for Section {percent} saved to {label_file_path}")

    print("All images processed and labeled.")