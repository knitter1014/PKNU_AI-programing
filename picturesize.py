from PIL import Image

# 이미지 파일 경로
image_path = './training/Clear_Sky.A_img1.png.25nrtv7n.ingestion-7c6b6d888b-fctbl.jpg'

# 이미지 열기
with Image.open(image_path) as img:
    # 이미지 크기 가져오기
    width, height = img.size
    print(f'이미지의 픽셀 크기: {width}x{height}')