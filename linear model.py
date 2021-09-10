# 对数据进行线性回归，即使用线性模型 y = ax + b 来拟合数据，此处 a 和 b 是待求的参数

# 使用 tape.gradient(ys, xs) 自动计算梯度；
# 使用 optimizer.apply_gradients(grads_and_vars) 自动更新模型参数。
# 使用 Keras（ tf.keras ）构建模型。Keras 是一个广为流行的高级神经网络 API，简单、快速而不失灵活性，现已得到 TensorFlow 的官方内置和全面支持。
# 我们没有显式地声明 a 和 b 两个变量并写出 y_pred = a * X + b 这一线性变换，而是建立了一个继承了 tf.keras.Model 的模型类 Linear 。
# 这个类在初始化部分实例化了一个 全连接层 （ tf.keras.layers.Dense ），并在 call 方法中对这个层进行调用，实现了线性变换的计算。


import tensorflow as tf
# import numpy as np

# 数据首先需要归一化
# X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
# y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

# X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
# y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())


X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class Linear(tf.keras.Model):  # model definition

    def __init__(self):  # constructor, initialize the layer of the model
        super().__init__()

        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)

        self.dense = tf.keras.layers.Dense(  # 全连接层
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, inputs):  # 模型调用，描述输入数据如何通过各种层而得到输出

        # 此处添加模型调用的代码（处理输入并返回输出），例如
        # x = layer1(input)
        # output = layer2(x)

        output = self.dense(inputs)
        return output


model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)  # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)  # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)
