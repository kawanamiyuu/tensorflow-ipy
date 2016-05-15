# -*- coding:utf-8 -*-
#訓練データとテストデータを読み込む
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#tensorflowをインポートしないと使えないよ
import tensorflow as tf

"""
入力層→中間層
入力層は784ユニット
中間層は100ユニット
活性化関数はReLUを使用する
"""
#順伝播までを定義する
#入力層を作るよ（ミニバッチ数は任意）
x = tf.placeholder(tf.float32, [None,784])

#入力層→中間層の重みを定義(重みの初期値をゼロにしては行けないのですよ！！)
W1 = tf.Variable(tf.truncated_normal([784,100],0,0.1))
#W1 = tf.Variable(tf.zeros([784,100]))

#中間層のバイアスを定義
b1 = tf.Variable(tf.zeros([100]))

#中間層の活性化関数（ReLU）を定義
y1 = tf.nn.relu(tf.matmul(x,W1) + b1)

"""
中間層→出力層
中間層は100ユニット
出力層は10ユニット
活性化関数はsoftmaxを使用する
"""
#中間層→出力層の重みを定義(重みの初期値をゼロにしてはいけないのですよ！！)
W2 = tf.Variable(tf.truncated_normal([100,10],0,0.1))
#W2 = tf.Variable(tf.zeros([100,10]))

#出力層のバイアスを定義
b2 = tf.Variable(tf.zeros([10]))

#出力層の活性化関数（softmax）を定義
y2 = tf.nn.softmax(tf.matmul(y1,W2) + b2)

#学習の処理を定義する
#今回用いる損失関数は交差エントロピー関数

#教師データを受け取る先の設定(ミニバッチ数は任意)
y_ = tf.placeholder(tf.float32,[None,10])

#交差エントロピー関数を定義する（多クラス分類だからね）
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y2),reduction_indices=[1]))

#勾配降下方を学習に用いる処理の定義(今回はAdamを使用している)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#変数の値を初期化する処理の定義
init = tf.initialize_all_variables()

#sessionの中で、学習を進められる
sess = tf.Session()
sess.run(init)

#モデルの評価を行う処理を定義する
#最大の値が同じ場所ならいいわけです
correct_prediction = tf.equal(tf.argmax(y2,1), tf.argmax(y_,1))

#正答率を計算する
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#学習を１０００回繰り返す
#ミニバッチのサイズは１００
for i in range(3000):
 batch_xs,batch_ys = mnist.train.next_batch(100)
 sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})

#100回学習したら、モデルの評価を行う
 if i % 100 == 0:
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print "finish"

