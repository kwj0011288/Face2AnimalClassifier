import cv2
import os

# 얼굴 감지기 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 이미지 폴더 경로 설정
base_dir = 'simple_images'
output_base_dir = 'crop_image'

# crop_image 폴더가 없으면 생성
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# 각 카테고리 폴더를 순회하며 이미지 처리
for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    
    if os.path.isdir(category_path):  # 폴더인지 확인
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            
            # 이미지 읽기
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 얼굴 감지
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                # 얼굴이 감지되면 얼굴 부분을 잘라서 저장
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        # 얼굴 부분을 잘라냄
                        face_crop = img[y:y+h, x:x+w]
                        
                        # 카테고리별 하위 폴더 경로 생성
                        category_output_path = os.path.join(output_base_dir, category)
                        if not os.path.exists(category_output_path):
                            os.makedirs(category_output_path)
                        
                        # 잘라낸 얼굴 이미지를 새로운 파일로 저장
                        output_path = os.path.join(category_output_path, 'cropped_' + img_name)
                        cv2.imwrite(output_path, face_crop)
                        print(f'Face cropped and saved: {output_path}')
                        break  # 여러 얼굴이 감지된 경우 첫 번째 얼굴만 사용
                else:
                    print(f'No face detected in {img_name} in category {category}')
