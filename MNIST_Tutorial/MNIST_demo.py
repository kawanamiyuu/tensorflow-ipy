# -*- coding:utf-8 -*-
#訓練データとテストデータを読み込む
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#tensorflowをインポートしないと使えないよ
import tensorflow as tf

#順伝播までを定義する
#入力層を作るよ（ミニバッチ数は任意）
x = tf.placeholder(tf.float32, [None,784])

#入力層→出力層間の重みを定義（＝入力層のノード数 * 出力層の数）
W = tf.Variable(tf.zeros([784,10]))

#バイアスを定義
b = tf.Variable(tf.zeros([10]))

#出力層の活性化関数を定義
y = tf.nn.softmax(tf.matmul(x,W) + b)

#学習の処理を定義する
#今回用いる損失関数は交差エントロピー関数

#教師データを受け取る先の設定(ミニバッチ数は任意)
y_ = tf.placeholder(tf.float32,[None,10])

#交差エントロピー関数を定義する（多クラス分類だからね）
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

#勾配降下方を学習に用いる処理の定義(今回の学習係数は0.5)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#変数の値を初期化する処理の定義
init = tf.initialize_all_variables()

#sessionの中で、学習を進められる
sess = tf.Session()
sess.run(init)

#学習を１０００回繰り返す
#ミニバッチのサイズは１００
for i in range(1000):
 batch_xs,batch_ys = mnist.train.next_batch(100)
 sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})

#モデルの評価を行う
#最大の値が同じ場所ならいいわけです
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#正答率を計算する
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#結果の出力を行う
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


