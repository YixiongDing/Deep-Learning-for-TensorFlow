import tensorflow as tf
import numpy as np


class MNISTLoader():

    def __init__(self):
        mnist = tf.keras.datasets.mnist  # 将从网络上自动下载 MNIST 数据集并加载
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()

        # MNIST中的图像默认为unit8 （0-255的数字），以下代码将其归一化为0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

        # 在 TensorFlow 中，图像数据集的一种典型表示是 [图像数目，长，宽，色彩通道数] 的四维张量。
        # 在上面的 DataLoader 类中， self.train_data 和 self.test_data 分别载入了 60,000 和 10,000 张大小为 28*28 的
        # 手写体数字图片。由于这里读入的是灰度图片，色彩通道数为 1（彩色 RGB 图像色彩通道数为 3），
        # 所以我们使用 np.expand_dims() 函数为图像数据手动在最后添加一维通道。

    def get_batch(self, batch_size):

        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]


# 卷积神经网络的一个示例实现如下所示，和 多层感知机(MLP) 在代码结构上很类似，只是新加入了一些卷积层和池化层。
# 这里的网络结构并不是唯一的，可以增加、删除或调整 CNN 的网络结构和参数，以达到更好的性能。

class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（valid 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)       # [batch_size, 28, 28, 32]
        x = self.pool1(x)            # [batch_size, 14, 14, 32]
        x = self.conv2(x)            # [batch_size, 14, 14, 64]
        x = self.pool2(x)            # [batch_size, 7, 7, 64]
        x = self.flatten(x)          # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)           # [batch_size, 1024]
        x = self.dense2(x)           # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


# 训练过程可视化
summary_writer = tf.summary.create_file_writer('./tensorboard')     # 参数为记录文件所保存的目录

# 接下来，当需要记录训练过程中的参数时，通过 with 语句指定希望使用的记录器，并对需要记录的参数（一般是 scalar）
# 运行 tf.summary.scalar(name, tensor, step=batch_index) ，即可将训练过程中参数在 step 时候的值记录下来。
# 这里的 step 参数可根据自己的需要自行制定，一般可设置为当前训练过程中的 batch 序号。


# 模型的训练

# 定义一些模型的超参数
num_epochs = 5
batch_size = 50
learning_rate = 0.001

# 实例化模型和数据读取类，并实例化一个 tf.keras.optimizer 的优化器（这里使用常用的 Adam 优化器）：
data_loader = MNISTLoader()
model = CNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 然后迭代进行以下步骤：
# 1. 从 DataLoader 中随机取一批训练数据；
# 2. 将这批数据送入模型，计算出模型的预测值；
# 3. 将模型预测值与真实值进行比较，计算损失函数（loss）。这里使用 tf.keras.losses 中的交叉熵函数作为损失函数；
# 4. 计算损失函数关于模型变量的导数；
# 5. 将求出的导数值传入优化器，使用优化器的 apply_gradients 方法更新模型参数以最小化损失函数。
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape()as tape:
        y_pred = model(X)

        # 在这里，我们没有显式地写出一个损失函数，而是使用了 tf.keras.losses 中的 sparse_categorical_crossentropy （交叉熵）函数，
        # 将模型的预测值 y_pred 与真实的标签值 y 作为函数参数传入，由 Keras 帮助我们计算损失函数的值.
        # 交叉熵作为损失函数，在分类问题中被广泛应用。其离散形式为 H(y, \hat{y}) = -\sum_{i=1}^{n}y_i \log(\hat{y_i}) ，
        # 其中 y 为真实概率分布， \hat{y} 为预测概率分布， n 为分类任务的类别个数。预测概率分布与真实分布越接近，则交叉熵的值越小，反之则越大。
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    # 每运行一次 tf.summary.scalar() ，记录器就会向记录文件中写入一条记录。除了最简单的标量（scalar）以外，TensorBoard 还可以对其他类型的
    # 数据（如图像，音频等）进行可视化，详见 TensorBoard 文档 。当我们要对训练过程可视化时，在代码目录打开终端（如需要的话进入 TensorFlow 的 conda 环境）
    # ，运行: tensorboard --logdir=./tensorboard
    # 然后使用浏览器访问命令行程序所输出的网址（一般是 http://name-of-your-computer:6006），即可访问 TensorBoard 的可视界面
    with summary_writer.as_default():
        tf.summary.scalar("loss", loss, step=batch_index)
        # tf.summary.scalar("MyScalar", my_scalar, step=batch_index)


# 模型的评估

# 我们使用 tf.keras.metrics 中的 SparseCategoricalAccuracy 评估器来评估模型在测试集上的性能，该评估器能够对模型预测的结果与真实结果进行比较，
# 并输出预测正确的样本数占总样本数的比例。我们迭代测试数据集，每次通过 update_state() 方法向评估器输入两个参数： y_pred 和 y_true ，
# 即模型预测出的结果和真实结果。评估器具有内部变量来保存当前评估指标相关的参数数值（例如当前已传入的累计样本数和当前预测正确的样本数）。
# 迭代结束后，我们使用 result() 方法输出最终的评估指标值（预测正确的样本数占总样本数的比例）。

# 我们实例化了一个 tf.keras.metrics.SparseCategoricalAccuracy 评估器，并使用 For 循环迭代分批次传入了测试集数据的预测结果与真实结果，
# 并输出训练后的模型在测试数据集上的准确率。
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy is: %f" % sparse_categorical_accuracy.result())
