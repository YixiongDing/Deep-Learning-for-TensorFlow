import tensorflow as tf
import tensorflow_datasets as tfds

# 使用 MobileNetV2 网络在 tf_flowers 五分类数据集上进行训练（为了代码的简短高效，在该示例中我们使用了 TensorFlow Datasets
# 和 tf.data 载入和预处理数据）。同时将 classes 设置为 5，对应于 5 分类的数据集。

# 定义一些模型的超参数
num_epochs = 5
batch_size = 50
learning_rate = 0.001

# 载入数据
dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label)).shuffle(1024).batch(batch_size)


# tf.keras.applications 中有一些预定义好的经典卷积神经网络结构，如 VGG16 、 VGG19 、 ResNet 、 MobileNet 等。
# 我们可以直接调用这些经典的卷积神经网络结构（甚至载入预训练的参数），而无需手动定义网络结构。
# model = tf.keras.applications.ResNet101()
# model = tf.keras.applications.ResNet101V2()
# model = tf.keras.applications.VGG19()
model = tf.keras.applications.MobileNetV2(weights=None, classes=5)

# 当执行以上代码时，TensorFlow 会自动从网络上下载 MobileNetV2 网络的预训练权值，因此在第一次执行代码时需要具备网络连接。
# 也可以通过将参数 weights 设置为 None 来随机初始化变量而不使用预训练权值。每个网络结构具有自己特定的详细参数设置，一些共通的常用参数如下：
# 1. input_shape ：输入张量的形状（不含第一维的 Batch），大多默认为 224 × 224 × 3 。一般而言，模型对输入张量的大小有下限，长和宽至少为 32 × 32 或 75 × 75 ；
# 2. include_top ：在网络的最后是否包含全连接层，默认为 True ；
# 3. weights ：预训练权值，默认为 'imagenet' ，即为当前模型载入在 ImageNet 数据集上预训练的权值。如需随机初始化变量可设为 None ；
# 4. classes ：分类数，默认为 1000。修改该参数需要 include_top 参数为 True 且 weights 参数为 None。

# 实例化一个 tf.keras.optimizer 的优化器（这里使用常用的 Adam 优化器）：
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 然后迭代进行以下步骤：
# 1. 从 dataset 中随机取一批训练数据；
# 2. 将这批数据送入模型，计算出模型的预测值；
# 3. 将模型预测值与真实值进行比较，计算损失函数（loss）。这里使用 tf.keras.losses 中的交叉熵函数作为损失函数；
# 4. 计算损失函数关于模型变量的导数；
# 5. 将求出的导数值传入优化器，使用优化器的 apply_gradients 方法更新模型参数以最小化损失函数。
for e in range(num_epochs):
    for images, labels in dataset:
        with tf.GradientTape() as tape:
            labels_pred = model(images, training=True)

            # 在这里，我们没有显式地写出一个损失函数，而是使用了 tf.keras.losses 中的 sparse_categorical_crossentropy （交叉熵）函数，
            # 将模型的预测值 labels_pred 与真实的标签值 labels 作为函数参数传入，由 Keras 帮助我们计算损失函数的值.
            # 交叉熵作为损失函数，在分类问题中被广泛应用。其离散形式为 H(y, \hat{y}) = -\sum_{i=1}^{n}y_i \log(\hat{y_i}) ，
            # 其中 y 为真实概率分布， \hat{y} 为预测概率分布， n 为分类任务的类别个数。预测概率分布与真实分布越接近，则交叉熵的值越小，反之则越大。
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=labels_pred)
            loss = tf.reduce_mean(loss)
            print("loss %f" % (loss.numpy()))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

    print(labels_pred)


# 模型的评估

# 我们使用 tf.keras.metrics 中的 SparseCategoricalAccuracy 评估器来评估模型在测试集上的性能，该评估器能够对模型预测的结果与真实结果进行比较，
# 并输出预测正确的样本数占总样本数的比例。我们迭代测试数据集，每次通过 update_state() 方法向评估器输入两个参数： y_pred 和 y_true ，
# 即模型预测出的结果和真实结果。评估器具有内部变量来保存当前评估指标相关的参数数值（例如当前已传入的累计样本数和当前预测正确的样本数）。
# 迭代结束后，我们使用 result() 方法输出最终的评估指标值（预测正确的样本数占总样本数的比例）。

# 我们实例化了一个 tf.keras.metrics.SparseCategoricalAccuracy 评估器，并使用 For 循环迭代分批次传入了测试集数据的预测结果与真实结果，
# 并输出训练后的模型在测试数据集上的准确率。
# sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
# num_batches = int(data_loader.num_test_data // batch_size)
# for batch_index in range(num_batches):
#     start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
#     y_pred = model.predict(data_loader.test_data[start_index: end_index])
#     sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
# print("test accuracy is: %f" % sparse_categorical_accuracy.result())
