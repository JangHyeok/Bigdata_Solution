%tensorflow_version 1.x

import tensorflow as tf
import numpy as np #배열을 위함

# 플레이스 홀더를 설정합니다.
X = tf.placeholder(tf.float32, shape=[None, 14])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([14, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# 가설을 설정합니다.
hypothesis = tf.matmul(X, W) + b

# 저장된 모델을 불러오는 객체를 선언합니다.
saver = tf.train.Saver()
model = tf.global_variables_initializer()

# 14가지 변수를 입력 받습니다.
avg_temper = float(input('평균기온(°C): '))
low_temper = float(input('최저기온(°C): '))
high_temper = float(input('최고 온도: '))
rainfall = float(input('일강수량(mm): '))
avg_ws = float(input('평균 풍속(m/s): '))
avg_rehumi = float(input('평균 상대습도(%): '))
total_sunhine = float(input('합계 일조시간(hr): '))
total_sunradi = float(input('합계 일사량(MJ/m2): '))
avg_landtemper = float(input('평균 지면온도(°C): '))
low_grasstemper = float(input('최저 초상온도(°C): '))
avg_humid = float(input('0.5M 평균 습도(%): '))
day_soilwater = float(input('10CM 일 토양수분(%): '))
shipments = float(input('출하량: '))
CPI = float(input('소비자 물가지수: '))

with tf.Session() as sess:
    sess.run(model)
    weight=20

    # 저장된 학습 모델을 파일로부터 불러옵니다.
    save_path = "/content/drive/My Drive/Colab Notebooks/선형회귀/save/saved.cpkt"
    saver.restore(sess, save_path)

    # 사용자의 입력 값을 이용해 배열을 만듭니다.
    data = ((avg_temper, low_temper, high_temper, rainfall, avg_ws, avg_rehumi, total_sunhine, total_sunradi, avg_landtemper, low_grasstemper, avg_humid, day_soilwater, shipments, CPI), )
    arr = np.array(data, dtype=np.float32)

    # 예측을 수행한 뒤에 그 결과를 출력합니다.
    x_data = arr[0:14]*weight
    dict = sess.run(hypothesis, feed_dict={X: x_data})
    print(dict[0])