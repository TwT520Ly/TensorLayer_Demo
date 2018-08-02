import tensorlayer as tl
import tensorflow as tf

sess = tf.InteractiveSession()

# 准备数据
X_train, Y_train, X_val, Y_val, X_test, Y_test = tl.files.load_mnist_dataset(shape=(-1, 784), path='data')

# 定义placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(tf.int64, shape=[None, ], name='y')

# 定义模型
network = tl.layers.InputLayer(inputs=x, name='input_layer')
network = tl.layers.DropoutLayer(layer=network, keep=0.8, name='drop_layer_1')
network = tl.layers.DenseLayer(layer=network, n_units=500, act=tf.nn.relu, name='relu_layer_1')
network = tl.layers.DropoutLayer(layer=network, keep=0.8, name='drop_layer_2')
network = tl.layers.DenseLayer(layer=network, n_units=500, act=tf.nn.relu, name='relu_layer_2')
network = tl.layers.DropoutLayer(layer=network, keep=0.5, name='drop_layer_3')
network = tl.layers.DenseLayer(layer=network, n_units=10, act=tf.identity, name='output_layer')

# 定义损失函数和评价指标
y_ = network.outputs
cost = tl.cost.cross_entropy(y_, y, name='cost')
# 对于每一个实例对应true或者是false
correct_prediction = tf.equal(tf.arg_max(y_, 1), y)
# true就是1， false就是0， 因此对所有的数据求平均，得到准确率。用于后面的输出结果。
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.arg_max(tf.nn.softmax(y_), 1)

# 定义优化器(使用所有参数进行训练)，利用交叉熵对参数进行迭代
train_params = network.all_params
train = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name='adam')\
    .minimize(cost, var_list=train_params)

# 初始化
tl.layers.initialize_global_variables(sess)

# 列出模型信息
network.print_params()
network.print_layers()

# 训练模型
tl.utils.fit(sess, network, train, cost, X_train, Y_train, x, y,
             acc=acc, batch_size=64, n_epoch=10, print_freq=5, X_val=X_val,
             y_val=Y_val, eval_train=False, tensorboard=True)

# 评估模型
tl.utils.test(sess, network, acc, X_test, Y_test, x, y, batch_size=None, cost=cost)

# 保存模型
tl.files.save_npz(network.all_params, name='model.npz')
sess.close()

