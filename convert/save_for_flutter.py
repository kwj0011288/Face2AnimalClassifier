import tensorflow as tf

# 저장된 모델 불러오기
model = tf.keras.models.load_model('./model.h5')

# TensorFlow Lite 모델로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# .tflite 파일로 저장
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite format and saved.")
