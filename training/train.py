import tensorflow as tf
import matplotlib.pyplot as plt

# 데이터 경로 설정
base_dir = 'crop_image'

# 데이터 제너레이터 설정 (데이터 증강 추가)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# 학습 데이터 제너레이터
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),  # 입력 크기를 사전 학습된 모델에 맞게 조정
    batch_size=32,
    color_mode='rgb',  # RGB로 변경
    class_mode='categorical',
    subset='training'
)

# 검증 데이터 제너레이터
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

validation_generator = validation_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation'
)

# 사전 학습된 모델 로드 (VGG16 사용)
base_model = tf.keras.applications.VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# 사전 학습된 층을 고정
base_model.trainable = False

# 새로운 모델 구성
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 요약 출력
model.summary()
# 모델 학습
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
)

# 사전 학습된 층을 풀기 (미세 조정)
base_model.trainable = True

# 특정 층부터 미세 조정
fine_tune_at = 10  # VGG16의 앞부분은 기본 특성을 추출하므로 일부만 훈련

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 모델 재컴파일 (더 낮은 학습률 사용)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 추가 학습 (미세 조정)
history_fine = model.fit(
    train_generator,
    epochs=40,
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
)

# 모델 평가
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Test accuracy: {test_acc:.2f}')

# 모델 저장
model.save('./model2.h5')

# 학습 결과 시각화
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 5))

# 1. 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training & validation accuracy')
plt.legend()

# 2. 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training & validation loss')
plt.legend()

plt.show()
