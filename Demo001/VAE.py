import tensorflow as tf
import tensorlayer as tl

# Denoising Autoencoder

# 创建会话
sess = tf.InteractiveSession()

# 准备数据
X_train, Y_train, X_val, Y_val, X_test, Y_test = tl.files.load_mnist_dataset(shape=(-1, 784), path='data/')

# 定义placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(tf.int64, shape=[None, ], name='y')

# 定义模型
network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout_layer')
network = tl.layers.DenseLayer(network, n_units=100, act=tf.nn.sigmoid, name='dense_layer')

recon_layer1 = tl.layers.ReconLayer(network, x_recon=x, n_units=784, name='recon_layer', act=tf.nn.sigmoid)

# 初始化
tl.layers.initialize_global_variables(sess)

network_params = network.all_params

# 预训练
recon_layer1.pretrain(sess, x, X_train, X_val, denoise_name='dropout_layer', n_epoch=100, save=False)

saver = tf.train.Saver()
saver.save(sess, 'model/VAE.ckpt')
sess.close()