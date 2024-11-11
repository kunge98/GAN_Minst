from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# from tensorflow import keras
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import sys


# 如果使用train参数运行则进入训练模式
TRAIN = False
if len(sys.argv) == 2 and sys.argv[1] == 'train':
    TRAIN = True

# 使用手写字体样本做训练
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
# 使用时尚单品样本做训练
# (train_images, _), (_, _) = keras.datasets.fashion_mnist.load_data()

# 因为卷积层的需求，增加色深维度
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# 规范化为-1 - +1
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# 图片生成模型
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# 原图、生成图辨别网络
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


generator = make_generator_model()

discriminator = make_discriminator_model()

# 随机生成一个向量，用于生成图片
noise = tf.random.normal([1, 100])

# 生成一张，此时模型未经训练，图片为噪点
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

# 判断结果
decision = discriminator(generated_image)

# 此时的结果应当应当趋近于0，表示为伪造图片
print('decision:', decision)

# 交叉熵损失函数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# 辨别模型损失函数
def discriminator_loss(real_output, fake_output):
    # 样本图希望结果趋近1
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # 自己生成的图希望结果趋近0
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # 总损失
    total_loss = real_loss + fake_loss
    return total_loss


# 生成模型的损失函数
def generator_loss(fake_output):
    # 生成模型期望最终的结果越来越接近1，也就是真实样本
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# 训练结果保存
checkpoint_dir = 'dcgan_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 1500
noise_dim = 100
num_examples_to_generate = 16

# 初始化16个种子向量，用于生成4x4的图片
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# @tf.function表示TensorFlow编译、缓存此函数，用于在训练中快速调用
@tf.function
def train_step(images):
    # 随机生成一个批次的种子向量
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 生成一个批次的图片
        generated_images = generator(noise, training=True)

        # 辨别一个批次的真实样本
        real_output = discriminator(images, training=True)
        # 辨别一个批次的生成图片
        fake_output = discriminator(generated_images, training=True)

        # 计算两个损失值
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # 根据损失值调整模型的权重参量
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # 计算出的参量应用到模型
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs+1):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # 每个训练批次生成一张图片作为阶段成功
        print("=======================================")
        generate_and_save_images(generator, epoch + 1, seed)

        # 每20次迭代保存一次训练数据
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


def generate_and_save_images(model, epoch, test_input):
    # 设置为非训练状态，生成一组图片
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    # 4格x4格拼接
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    # 保存为png
    plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close()


# 遍历所有png图片，汇总为gif动图
def write_gif():
    anim_file = 'dcgan.gif'
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


# 生成一张初始状态的4格图片，应当是噪点
generate_and_save_images(generator, 0000, seed)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 使用 GPU 0，1
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用 GPU 0，1
# # GPU settings
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
# if TRAIN:
#     # 以训练模式运行，进入训练状态
#     train(train_dataset, EPOCHS)
#     write_gif()
# else:
#     # 非训练模式，恢复训练数据
#     checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#     print("After training:")
#     # 显示训练完成后，生成图片的辨别结果
#     generated_image = generator(noise, training=False)
#     decision = discriminator(generated_image)
#     # 结果应当趋近1
#     print('decision:', decision)
#     # 重新生成随机值，生成一组图片保存
#     seed = tf.random.normal([num_examples_to_generate, noise_dim])
#     generate_and_save_images(generator, 6666, seed)


train(train_dataset, EPOCHS)