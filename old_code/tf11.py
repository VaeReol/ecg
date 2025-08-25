import tensorflow as tf

# # 假设我们有输入数据 x，维度为 [batch_size, input_dim]
# # 目标是实现一个全连接层，将输入的维度从 input_dim 转换为 output_dim

# def fully_connected(x, input_dim, output_dim, activation_fn=None):
#     # 1. 初始化权重和偏置
#     # 使用 tf.random_normal 初始化权重（均值为0，标准差为1）
#     weights = tf.Variable(tf.random_normal([input_dim, output_dim]))
#     # 偏置初始化为0
#     biases = tf.Variable(tf.zeros([output_dim]))
    
#     # 2. 前向传播：x * W + b
#     z = tf.matmul(x, weights) + biases
    
#     # 3. 应用激活函数（如果提供的话）
#     if activation_fn:
#         return activation_fn(z)
#     else:
#         return z

# # 使用示例

# # 假设输入为 10 个样本，每个样本有 5 个特征
# batch_size = 10
# input_dim = 5
# output_dim = 3

# # 随机生成输入数据
# x = tf.placeholder(tf.float32, [None, input_dim])

# # 创建全连接层，使用 ReLU 激活函数
# output = fully_connected(x, input_dim, output_dim, activation_fn=tf.nn.relu)

# # 创建一个会话并初始化变量
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
    
#     # 假设我们有一批输入数据
#     input_data = [[1, 2, 3, 4, 5]] * batch_size
    
#     # 计算输出
#     result = sess.run(output, feed_dict={x: input_data})
    
#     print("Output of fully connected layer:")
#     print(result)
import tensorflow as tf

def full_connection(x, input_dim, output_dim, activation_fn=None):
    w = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.zeros([output_dim]))
    output = tf.matmul(x, w) + b
    if activation_fn:
        output = activation_fn(output)
    return output


with tf.Session() as sess:
    input_data = sess.run(tf.random_normal([64, 5])) 
    input_dim, output_dim = 5, 3
    result = full_connection(input_data, input_dim, output_dim, activation_fn=tf.nn.relu)
    sess.run(tf.global_variables_initializer())
    result = sess.run(result)
    print(result.shape)