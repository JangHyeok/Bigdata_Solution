%tensorflow_version 1.x

import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv

model = tf.global_variables_initializer();

data = read_csv('/content/drive/My Drive/Colab Notebooks/선형회귀/csv/HqPriceV2.csv', sep=',')

xy = np.array(data, dtype=np.float32)

# 4->8개의 변인을 입력을 받습니다.
x_data = xy[:, 1:-1]

# 가격 값을 입력 받습니다.
y_data = xy[:, [-1]]

# 플레이스 홀더를 설정합니다.
X = tf.placeholder(tf.float32, shape=[None, 14])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([14, 1]), name="weight") #가중치값 초기화
b = tf.Variable(tf.random_normal([1]), name="bias") #바이어스값 초기화

# 가설을 설정합니다.
hypothesis = tf.matmul(X, W) + b

# 비용 함수를 설정합니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 최적화 함수를 설정합니다.
#학습률 적절하게 하는것이 중요 0.000005~0.000000000005)
#00000000005=NaN
#000000000005=500
#000000000006=700
#0000000000005=360
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000000000005)
train = optimizer.minimize(cost)

# 세션을 생성합니다.
sess = tf.Session()

# 글로벌 변수를 초기화합니다.
sess.run(tf.global_variables_initializer())

# 학습을 수행합니다.
#100001
for step in range(100001):
    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 500 == 0:
        print("step#", step, "cost : ", cost_)
        print("- hypo : ", hypo_[0])

# 학습된 모델을 저장합니다.
saver = tf.train.Saver()
save_path = saver.save(sess, "/content/drive/My Drive/Colab Notebooks/선형회귀/save/saved.cpkt")
print('학습된 모델을 저장했습니다.')