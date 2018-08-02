import tensorlayer as tl
import tensorflow as tf

# 创建参数
batch_size = 64
n_epoch = 100
learning_rate = 0.001
print_freq = 10

# 创建会话
sess = tf.InteractiveSession()

# 加载数据
X_train, Y_train, X_val, Y_val, X_test, Y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1), path='data/')

# 创建placeholder
x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1], name='x')
y = tf.placeholder(tf.int64, shape=[batch_size, ], name='y')

network = tl.layers.InputLayer(x, name='input_layer')
# 构建模型(expert)

# 构建模型(simple)
network = tl.layers.Conv2d(network, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2d_layer_1')
network = tl.layers.MaxPool2d(network, filter_size=(2, 2), padding='SAME', name='maxpool_layer_1')
network = tl.layers.Conv2d(network, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2d_layer_2')
network = tl.layers.MaxPool2d(network, filter_size=(2, 2), padding='SAME', name='maxpool_layer_2')

# end of network
network = tl.layers.FlattenLayer(network, name='flatten_layer')
network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout_layer_1')
network = tl.layers.DenseLayer(network, n_units=256, act=tf.nn.relu, name='dense_layer_1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout_layer_2')
network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output')

# 计算损失函数
y_ = network.outputs
cost = tl.cost.cross_entropy(y_, y, name='cost')
correct_prediction = tf.equal(tf.arg_max(y_, 1), y)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练过程
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=train_params)

# 初始化网络参数，输出网络参数和网络结构
tl.layers.initialize_global_variables(sess)
network.print_params()
network.print_layers()

print("Learning rate: ", learning_rate)
print("")

