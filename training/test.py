import tensorflow as tf
import numpy as np

# 저장된 모델 불러오기
model = tf.keras.models.load_model('model2.h5')

# 입력 이미지 경로
image_path = './tim.jpg'  # 실제 이미지 파일 경로로 변경

# 모델의 출력 클래스 수 확인
num_classes = model.output_shape[-1]

# 클래스 이름 동적으로 생성 (예: class_0, class_1, ...)
class_names = {i: f'class_{i}' for i in range(num_classes)}

# 사용자 지정 클래스 이름이 있다면 업데이트 (옵션)
custom_class_names = {
    0: 'bear',
    1: 'cat',
    2: 'dog',
    3: 'rabbit',
    4: 'turtle'
}
class_names.update(custom_class_names)

# 이미지 불러오기 및 전처리
img = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=(224, 224))  # RGB로 설정
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0  # 스케일링
img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

# 예측
prediction = model.predict(img_array)

# 모든 클래스에 대한 확률 출력
print("Class probabilities:")
for idx, prob in enumerate(prediction[0]):
    print(f"{class_names.get(idx, f'Unknown class {idx}')}: {prob:.2%}")

# 예측된 클래스 출력
predicted_class = np.argmax(prediction, axis=1)
print(f'\nPredicted class: {class_names.get(predicted_class[0], "Unknown class")}')
