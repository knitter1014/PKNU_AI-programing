import requests
import os

ACCESS_KEY = "29Yj8PFRYf_5K3UogCgue1BU5gUDVjxjYx-GIoujZHI"  # Unsplash API Access Key
query = "sky"  # 검색어
images_to_download = 50  # 다운로드할 이미지 수
per_page = 30  # 한 페이지당 가져올 이미지 수 (최대 30)
pages = images_to_download // per_page  # 필요한 페이지 수

# 이미지 저장 디렉토리 생성
output_dir = "sky_images"
os.makedirs(output_dir, exist_ok=True)

downloaded_images = 0

# 페이지별로 데이터 요청
for page in range(1, pages + 1):
    url = f"https://api.unsplash.com/search/photos?query={query}&client_id={ACCESS_KEY}&page={page}&per_page={per_page}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        for result in data["results"]:
            image_url = result["urls"]["regular"]
            image_id = result["id"]
            
            # 이미지 다운로드
            img_data = requests.get(image_url).content
            with open(f"{output_dir}/{image_id}.jpg", "wb") as img_file:
                img_file.write(img_data)
                downloaded_images += 1

            print(f"Downloaded {downloaded_images}/{images_to_download} images")

            # 1500장을 다 다운로드하면 종료
            if downloaded_images >= images_to_download:
                break
    else:
        print(f"Error: Unable to fetch page {page}, Status Code: {response.status_code}")
        break