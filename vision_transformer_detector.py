"""Build a Vision Transformer detector in TensorFlow 2.9
基于 2021 年 6 月的第二版 Vision Transformer： https://arxiv.org/abs/2010.11929
"""

from collections.abc import Iterable
from enum import Enum
import sys

import PIL
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras


# 使用 Enum 创建模块常量 module constants。
# 根据 Google Python Style Guide，对模块常量使用大写字母。
class Constants(Enum):
    CLASSES = 80  # 如果使用 COCO 数据集，则需要探测 80 个类别。
    # MODEL_IMAGE_SIZE 格式为 height, width。如果显存不够，可以使用 304 大小。
    MODEL_IMAGE_SIZE = 608, 608  # 304, 304

    EPSILON = 1e-8  # 进行除法计算时，给分母中加上一个极小量，避免除法出错。

    # MAX_DETECT_OBJECTS_QUANTITY 是在每个图片中，最多能够探测到的物体数量。
    # 前 16 张图片，一张图片内的最大标注数量为 16。前 80 张图片，最大标注数量为 42。
    MAX_DETECT_OBJECTS_QUANTITY = 17

    # LATEST_RELATED_IMAGES: 一个整数，表示最多使用多少张相关图片来计算一个类别的 AP。
    # 在前 80 张图片中，类别 'person' 出现次数最多，为 40 次。
    LATEST_RELATED_IMAGES = 3  # 41

    # BBOXES_PER_IMAGE: 一个整数，表示对于一个类别的每张相关图片，最多使用
    # BBOXES_PER_IMAGE 个 bboxes 来计算 AP。
    # 在前 80 张图片的每一张图片内，类别 'person' 的标注数量最多，为 14 个。
    BBOXES_PER_IMAGE = 14  # 15

    # 指标 MeanAveragePrecision 用到的模块常量，使用大写字母。
    # OBJECTNESS_THRESHOLD: 一个浮点数，表示物体框内，是否存在物体的置信度阈值。
    OBJECTNESS_THRESHOLD = 0.5
    # CLASSIFICATION_CONFIDENCE_THRESHOLD: 一个浮点数，表示物体框的类别置信度阈值。
    CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5


def check_inf_nan(inputs, name, max_value=50000,
                  replace_nan: None | float = None):
    """检查输入中是否存在 inf 和 NaN 值，并进行提示。

    Arguments:
        inputs: 一个数据类型的张量，可以是任意形状。
        name: 一个字符串，是输入张量的名字。
        max_value: 一个整数，如果当前输入张量的最大值，大于 max_value ，则打印输出
            当前输入张量的最大值。尤其注意数值在超过 50,000 之后，是有可能无法使用混合精
            度计算的，因为 float16 格式下，数值达到 65520 时就会产生 inf 值。
        replace_nan: 一个浮点数或者 None。如果是浮点数，则把输入中的 NaN 值替换为该浮
            点数。如果为 None，则不对 NaN进行处理。
    """

    if type(inputs) != tuple:

        # 排除整数类型和浮点数类型，只使用张量和数组。
        if not isinstance(inputs, (int, float)):

            input_is_keras_tensor = keras.backend.is_keras_tensor(inputs)
            # 如果输入是符号张量 symbolic tensor，则不检查该张量。
            if not input_is_keras_tensor:
                input_inf = tf.math.is_inf(inputs)
                if tf.math.reduce_any(input_inf):
                    # 在图模式下运行时，print 不起作用，只能用 tf.print
                    tf.print(f'\nInf! Found in {name}, its shape: ',
                             input_inf.shape)

                input_nan = tf.math.is_nan(inputs)
                if tf.math.reduce_any(input_nan):
                    tf.print(f'\nNaN! Found in {name}, its shape: ',
                             input_nan.shape)
                    if replace_nan is not None:
                        # 把 NaN 替换为浮点数 replace_nan。
                        inputs = tf.where(input_nan, replace_nan, inputs)

                current_max = tf.math.reduce_max(inputs)
                if current_max > max_value:
                    max_value = current_max
                    tf.print(f'\nIn {name}, its shape: ', inputs.shape)
                    tf.print(f'max_value: ', max_value)

    else:
        # 模型的输出值，将进入这个分支。

        for i, each_input in enumerate(inputs):

            input_is_keras_tensor = keras.backend.is_keras_tensor(each_input)
            # 如果输入是符号张量 symbolic tensor，则不检查该张量。
            if not input_is_keras_tensor:
                subname = f'{name}_{i}'
                input_inf = tf.math.is_inf(each_input)
                if tf.math.reduce_any(input_inf):
                    tf.print(f'\nInf! Found found in {subname}, its shape: ',
                             input_inf.shape)

                input_nan = tf.math.is_nan(each_input)
                if tf.math.reduce_any(input_nan):
                    tf.print(f'\nNaN! Found in {subname}, its shape: ',
                             input_nan.shape)
                    if replace_nan is not None:
                        # 把 NaN 替换为浮点数 replace_nan。
                        each_input = tf.where(input_nan,
                                              replace_nan, each_input)

                current_max = tf.math.reduce_max(each_input)
                if current_max > max_value:
                    max_value = current_max
                    tf.print(f'\nIn {subname}, its shape: ', each_input.shape)
                    tf.print(f'max_value change to: ', max_value)
    return inputs


class MishActivation(keras.layers.Layer):
    """把 tfa 模块的 mish 激活函数封装到 keras.layers.Layer 中。
    为了便于迁移 serialization ，使用子类方法 subclassing，不使用层 layers.Lambda。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # noinspection PyMethodMayBeStatic
    def call(self, inputs):
        return tfa.activations.mish(inputs)


class PositionEncoding(keras.layers.Layer):
    """因为进行位置编码 position encoding 时用到了 tf.range，必须将其封装到一个
    layers.Layer 里面，才能用 plot_model 画出模型结构图，否则会报错:
    eager tensor 没有 "_keras_history" 属性。
    """

    def __init__(self, patches_quantity, embedding_dim,
                 max_weight_norm, **kwargs):
        super().__init__(**kwargs)
        self.patches_quantity = patches_quantity
        self.embedding_dim = embedding_dim
        self.max_weight_norm = max_weight_norm

        # embedding 层的权重是 sparse tensor，所以无法使用 keras.constraints。参见
        # Keras 记录的 issue： https://github.com/keras-team/keras/issues/15818
        # 经多次实验发现，在指标 AP 接近 100%，该权重会出现 NaN。
        self.position_embeddings = keras.layers.Embedding(
            input_dim=patches_quantity, output_dim=self.embedding_dim,
            # embeddings_constraint=weights_constraint,  # 无法使用权重约束。
            name='position_embedding')

    # noinspection PyUnusedLocal
    def call(self, inputs):
        # positions 形状为 (41209,)。
        positions = tf.range(self.patches_quantity)
        # positions 形状为 (1, 41209)。必须用 tf.newaxis 转为 2D 张量，后续给
        # Embedding 层才能得到 3D 张量。如果不用 tf.newaxis，最后 image_patches 的
        # 第 0 维度大小，将会从 None 变为一个整数，导致模型错误。
        position_encoding = positions[tf.newaxis, :]

        # positions 形状为 (1, 41209, output_dim)。
        embedded_positions = self.position_embeddings(position_encoding)

        return embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            'patches_quantity': self.patches_quantity,
            'embedding_dim': self.embedding_dim,
            'max_weight_norm': self.max_weight_norm,
        })
        return config


class ExtractImagePatches(keras.layers.Layer):
    """将 extract_patches 封装到一个 layers.Layer 里面。

    使用说明：
        因为使用了 tf.image.extract_patches 的原因，只能保存模型的权重，无法直接保存模型。
        这可能是 TF 本身的 bug 导致的。
    """

    def __init__(self, patch_sizes, **kwargs):
        super().__init__(**kwargs)
        self.patch_sizes = patch_sizes

    def call(self, inputs):
        # patches 形状为 (batch_size, ceil(*MODEL_IMAGE_SIZE/3), 27)，这是因为
        # 设置了 padding='SAME'。为了获得原图边缘的像素点，所以设置 padding='SAME'。
        # 最后一个维度为 27，是因为 patch 大小为 3x3=9，加上 3 个通道，所以是 27。如果
        # MODEL_IMAGE_SIZE = 608, 608，则 patches 形状为
        # (batch_size, 203, 203, 27)。
        patches = tf.image.extract_patches(
            images=inputs, sizes=self.patch_sizes, strides=self.patch_sizes,
            rates=[1, 1, 1, 1], padding='SAME')

        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_sizes': self.patch_sizes,
        })
        return config


class ClipWeight(keras.constraints.Constraint):
    """限制权重在一个固定的范围，避免出现过大权重和 NaN。

    Attributes:
        min_weight: 一个浮点数，是权重的最小值。
        max_weight: 一个浮点数，是权重的最大值。
    """

    def __init__(self, max_weight):
        self.min_weight = -max_weight
        self.max_weight = max_weight

    def __call__(self, w):
        nan_weight = tf.math.is_nan(w)  # 找出权重为 NaN 的位置。
        # 把 NaN 权重替换为数值 1.
        w = tf.cond(tf.reduce_any(nan_weight),
                    true_fn=lambda: tf.where(nan_weight, 1.0, w),
                    false_fn=lambda: w)

        return tf.clip_by_value(w, clip_value_min=self.min_weight,
                                clip_value_max=self.max_weight)

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_weight': self.max_weight,
        })
        return config


def transformer_preprocessor(inputs, patch_size,
                             embedding_dim, max_weight, clip_weight):
    """把输入图片进行预处理，包括 2 部分：把图片分割成小块 patches，然后给 patches 加上
    positional encoding。

    Arguments：
        inputs：一个 4D 图片张量，形状为 (batch_size, *MODEL_IMAGE_SIZE, 3)，数据类
            型为 tf.float32。可以用全局变量 MODEL_IMAGE_SIZE 设置不同大小的图片输入。
            为了方便说明，下面以 MODEL_IMAGE_SIZE = 608, 608 为例。
        patch_size：一个整数，代表的是将图片分割成小方块之后，每个小方块的边长。
            为了方便说明，下面以 patch_size = 3, 3 为例。
        embedding_dim：一个整数，代表的是将图片预处理之后，输出张量最后一个维度的大小。
        max_weight： 一个浮点数，用于设置权重的最大值。
        clip_weight： 一个布尔值，如果为 True，则直接对权重使用 clip_by_value，否则用
            最大范数 MaxNorm 限制权重。
    Returns:
        image_patches: 一个 tf.float32 类型的张量，张量形状为
            (batch_size, patches_quantity, embedding_dim)。
    """

    # 限制权重。避免过大的权重导致的 NaN 问题。
    if clip_weight:
        weights_constraint = ClipWeight(max_weight)
    else:
        weights_constraint = keras.constraints.MaxNorm(
            max_value=max_weight, axis=0)

    patch_sizes = [1, patch_size, patch_size, 1]

    # 下面使用 noinspection PyCallingNonCallable，是因为 TF 的版本问题，导致
    # Pycharm 无法识别 keras.layers.Layer，会出现调用报错，手动关闭此报错即可。
    # noinspection PyCallingNonCallable
    patches = ExtractImagePatches(
        patch_sizes=patch_sizes, name='split_image_into_patches')(inputs)

    sequence_length = patches.shape[-1]

    # patches 形状为 (batch_size, 1296, 867)，将其转为 3D 张量。取决于 patch_sizes。
    # 创建 Keras 模型时，必须使用 keras.layers.Reshape，不能使用 tf.reshape，这是因为
    # tf.reshape 不能处理 symbolic 张量（即 Keras 张量，其第 0 维度大小为 None）。
    patches = keras.layers.Reshape(
        target_shape=(-1, sequence_length), name='flatten_patches')(patches)

    # 注意下面不能用 tf.shape，而是要用其属性 tensor.shape。
    # 这是因为对符号张量 keras tensors，tf.shape 会返回符号张量的形状(实际上是阶数)，而
    # 不是更具体的形状信息，关于这一点，可以参见 tf.shape 的官方文档。
    patches_quantity = patches.shape[1]

    # 下面的 positional encoding，其实不使用也不会有太大影响。
    # 因为 TF 的版本问题，导致 Pycharm 无法识别 keras.layers.Layer，会出现调用报错，
    # 使用 noinspection PyCallingNonCallable 手动关闭此报错即可。
    # noinspection PyCallingNonCallable
    embedded_positions = PositionEncoding(
        patches_quantity=patches_quantity, embedding_dim=1,
        max_weight_norm=max_weight)(patches)

    # linear_projection 形状为 (batch_size, 1296, 28)，是对 patches
    # 进行线性变换。这里假设 embedding_dim=28。
    linear_projection = keras.layers.Dense(
        units=embedding_dim,
        kernel_constraint=weights_constraint,
        bias_constraint=weights_constraint,
        name='linear_projection')(patches)

    # image_patches 形状为 (None, 1296, 28)。这里假设 embedding_dim=28。
    # 注意如果用大写的 Add，则需要按照类的方法处理，即生成个体后再传入参数。
    image_patches = keras.layers.add(
        [linear_projection, embedded_positions],
        name='embedded_patches')

    return image_patches


def transformer_encoder(embedded_image_patches, use_mish,
                        num_heads, key_dim, dropout, mlp_quantities,
                        repeat_times, max_weight, clip_weight,
                        training=None):
    """Vision Transformer 模型的 encoder 部分。

    Arguments：
        embedded_image_patches：一个 tf.float32 类型的张量，张量形状为
            (batch_size, patches_quantity, output_dim)。
        use_mish：一个布尔值，如果为 True，则使用 mish 激活函数。
        num_heads：一个整数，是 MultiHeadAttention 层的 heads 数量。
        key_dim：一个整数，是 MultiHeadAttention 层的参数，是一个计算
            MultiHeadAttention 的中间变量。
        dropout：一个浮点数，是 MultiHeadAttention 层内 dropout 的比例值。
        mlp_quantities：一个整数，是 encoder 中 MLP 的数量。
        repeat_times：一个整数，是 encoder 内所有操作进行重复的次数。
        max_weight： 一个浮点数，用于设置权重的最大值。
        clip_weight： 一个布尔值，如果为 True，则直接对权重使用 clip_by_value，否则用
            最大范数 MaxNorm 限制权重。
        training: 一个布尔值，用于设置模型是处在训练模式或是推理 inference 模式。
            在预测时，如果不使用 predict 方法，而是直接调用模型的个体，则必须设置参
            数 training=False，比如 model(x, training=False)。因为这样才能让模
            型的 dropout 层和 BatchNormalization 层以 inference 模式运行。而如
            果是使用 predict 方法，则不需要设置该 training 参数。
    Returns:
        x: 一个 tf.float32 类型的张量，形状和输入张量 embedded_image_patches 相同。
    """

    # 限制权重。避免过大的权重导致的 NaN 问题。
    if clip_weight:
        weights_constraint = ClipWeight(max_weight)
    else:
        weights_constraint = keras.constraints.MaxNorm(
            max_value=max_weight, axis=0)

    # 因为要在 for 循环内反复使用，所以要借助一个变量 x。
    x = embedded_image_patches

    for i in range(1, repeat_times + 1):
        # 设置 side_connection_1，在 MultiHeadAttention 之后要加回去。
        side_connection_1 = x
        x = keras.layers.LayerNormalization(
            axis=-1,
            beta_constraint=weights_constraint,
            gamma_constraint=weights_constraint,
        )(x)

        # 因为 MultiHeadAttention 的 dropout 参数不接受 None，所以改用数值 0。
        if dropout is None:
            dropout_attention = 0
        else:
            dropout_attention = dropout
        x = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            dropout=dropout_attention,
            kernel_constraint=weights_constraint,
            bias_constraint=weights_constraint,
        )(query=x, value=x, training=training)

        x = keras.layers.add([x, side_connection_1],
                             name=f'residual_connection_{i}_1')

        side_connection_2 = x
        x = keras.layers.LayerNormalization(
            axis=-1,
            beta_constraint=weights_constraint,
            gamma_constraint=weights_constraint,
        )(x)

        # last_dimensionality 是输入张量最后一个维度的大小。
        last_dimensionality = x.shape[-1]

        # mlp_units 是一个数组，从大到小排列，表示各个 Dense 层的单元数逐层减小。
        mlp_units = last_dimensionality * 2 ** np.arange(
            start=(mlp_quantities - 1), stop=-1, step=-1)

        for j in range(mlp_quantities):
            x = keras.layers.Dense(
                units=mlp_units[j],
                kernel_constraint=weights_constraint,
                bias_constraint=weights_constraint,
                name=f'MLP_{i}_{j + 1}'
            )(x)

            if use_mish:
                # noinspection PyCallingNonCallable
                x = MishActivation()(x)
                pass
            else:
                # noinspection PyCallingNonCallable
                x = tfa.layers.GELU()(x)

            if dropout is not None:
                x = keras.layers.Dropout(rate=dropout)(x, training=training)

        # 给 encoder 的输出结果加上名字 encoded_images。
        if i == repeat_times:
            x = keras.layers.add([x, side_connection_2], name='encoded_images')
        else:
            x = keras.layers.add([x, side_connection_2],
                                 name=f'residual_connection_{i}_2')

    return x


def mlp_head(encoder_outputs, use_mish,
             mlp_head_last_units, dense_layers_quantity,
             dense_mish_block_repeats,
             dropout, max_weight, clip_weight, training=None):
    """创建一个 Vision Transformer 物体探测器。

    Arguments：
        encoder_outputs：一个 tf.float32 型张量，形状为 (batch_size, patches,
            embedding_dim)，是 encoder 部分的输出。
        use_mish：一个布尔值，如果为 True，则使用 mish 激活函数。
        mlp_head_last_units：一个整数，是模型 MLP Head 部分，最后一个 Dense 层
            的单元数量。
        dense_layers_quantity：一个整数，是 MLP Head 部分，Dense 层数量。
        dense_mish_block_repeats：一个整数，是把 Dense 层和 mish 层作为一个 block，
            对该 block 重复的次数。
        dropout：一个浮点数，是 dropout 的比例值。
        max_weight： 一个浮点数，用于设置权重的最大值。
        clip_weight： 一个布尔值，如果为 True，则直接对权重使用 clip_by_value，否则用
            最大范数 MaxNorm 限制权重。
        training: 一个布尔值，用于设置模型是处在训练模式或是推理 inference 模式。
            在预测时，如果不使用 predict 方法，而是直接调用模型的个体，则必须设置参
            数 training=False，比如 model(x, training=False)。因为这样才能让模
            型的 dropout 层和 BatchNormalization 层以 inference 模式运行。而如
            果是使用 predict 方法，则不需要设置该 training 参数。
    Returns:
        mlp_head_outputs: 一个 tf.float32 类型的张量，张量形状为
            (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。
    """

    # 限制权重。避免过大的权重导致的 NaN 问题。
    if clip_weight:
        weights_constraint = ClipWeight(max_weight)
    else:
        weights_constraint = keras.constraints.MaxNorm(
            max_value=max_weight, axis=0)

    # x 形状为 (batch_size, patches, MAX_DETECT_OBJECTS_QUANTITY)
    x = keras.layers.Dense(
            units=Constants.MAX_DETECT_OBJECTS_QUANTITY.value,
            kernel_constraint=weights_constraint,
            bias_constraint=weights_constraint,
            )(encoder_outputs)

    # x 形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY, patches)
    x = keras.layers.Reshape(
        target_shape=(Constants.MAX_DETECT_OBJECTS_QUANTITY.value, -1),
    )(x)

    dense_layer_units = mlp_head_last_units * 2 ** np.arange(
        dense_layers_quantity)

    for units in reversed(dense_layer_units):
        # 把 Dense 层、 mish 层和 dropout 层作为一个 block，对该 block 重复若干次。
        for _ in range(dense_mish_block_repeats):
            # x 形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY, units)。
            x = keras.layers.Dense(
                units=units,
                kernel_constraint=weights_constraint,
                bias_constraint=weights_constraint,
                )(x)

            if use_mish:
                # noinspection PyCallingNonCallable
                x = MishActivation()(x)
            else:
                # noinspection PyCallingNonCallable
                x = tfa.layers.GELU()(x)

            if dropout is not None:
                x = keras.layers.Dropout(rate=dropout)(x, training=training)

    # mlp_head_outputs 形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。
    mlp_head_outputs = keras.layers.Dense(
        units=6,
        kernel_constraint=weights_constraint,
        bias_constraint=weights_constraint,
        name=f'MLP_Head_no_Sigmoid')(x)

    return mlp_head_outputs


def create_vision_transformer_detector(
        input_shape=None, patch_size=17, embedding_dim=28,
        encoder_num_heads=8, encoder_key_dim=40, dropout=None,
        encoder_mlp_quantities=8,
        encoder_repeat_times=8,
        mlp_head_last_units=136, mlp_head_dense_layers_quantity=7,
        mlp_head_dense_mish_block_repeats=1,
        use_mish=True,
        max_weight=10, clip_weight=True, training=None):
    """创建一个 Vision Transformer 物体探测器。

    Arguments：
        input_shape：一个 3D 图片张量的形状，形状可以为 (608, 608, 3)，数据类型为
            tf.float32。可以用全局变量 MODEL_IMAGE_SIZE 设置不同大小的图片输入。
            为了方便说明，下面的各个张量形状均假设 input_shape 为 (608, 608, 3)。
        patch_size：一个整数，代表的是将图片分割成小方块之后，每个小方块的边长。
        embedding_dim：一个整数，代表的是将图片预处理之后，输出张量最后一个维度的大小。
        encoder_num_heads：一个整数，是 MultiHeadAttention 层的 heads 数量。
        encoder_key_dim：一个整数，是 MultiHeadAttention 层的参数，是一个计算
            MultiHeadAttention 的中间变量。
        dropout：一个浮点数，是 MultiHeadAttention 部分，以及 MLP Head 部分 dropout
            的比例值。
        encoder_mlp_quantities：一个整数，是 encoder 中 MLP 的数量。
        encoder_repeat_times：一个整数，是 encoder 内所有操作进行重复的次数。
        mlp_head_last_units：一个整数，是模型 MLP Head 部分，最后一个 Dense 层
            的单元数量。
        mlp_head_dense_layers_quantity：一个整数，是 MLP Head 部分，Dense 层数量。
        mlp_head_dense_mish_block_repeats：一个整数，是 MLP Head 部分，把 Dense 层
            和 mish 层作为一个 block，对该 block 重复的次数。
        use_mish：一个布尔值，如果为 True，则使用 mish 激活函数。
        max_weight： 一个浮点数，用于设置权重的最大值。
        clip_weight： 一个布尔值，如果为 True，则直接对权重使用 clip_by_value，否则用
            最大范数 MaxNorm 限制权重。
        training: 一个布尔值，用于设置模型是处在训练模式或是推理 inference 模式。
            在预测时，如果不使用 predict 方法，而是直接调用模型的个体，则必须设置参
            数 training=False，比如 model(x, training=False)。因为这样才能让模
            型的 dropout 层和 BatchNormalization 层以 inference 模式运行。而如
            果是使用 predict 方法，则不需要设置该 training 参数。
    Returns:
        model: 一个 Keras 模型，模型输出一个 tf.float32 类型的张量，张量形状为
            (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。
            最后 1 个维度大小为 6，表示每个预测结果是一个长度为 6 的向量，每一位的数值都
            是 [0, 1] 的范围。各自代表的含义是：
            第 0 位：是 objectness，即物体框内是否有物体的置信度。
            第 1 位：代表 80 个类别。
            最后 4 位：是预测框的位置和坐标，格式为 (x, y, height, width)，其中 x，y
            是预测框的中心点坐标，height, width 是预测框的高度和宽度。这 4 个数值的范围都
            在 [0, 1] 之间，是一个比例值，表示在原图中的比例大小。
    """

    keras.backend.clear_session()

    if input_shape is None:
        input_shape = *Constants.MODEL_IMAGE_SIZE.value, 3

    # inputs 是一个 Keras tensor，也叫符号张量 symbolic tensor，这种张量没有实际的值，
    # 只是在创建模型的第一步————构建计算图时会用到，模型创建好之后就不再使用符号张量。
    image_inputs = keras.Input(shape=input_shape, name='images')

    # embedded_image_patches 形状为 (batch_size, 1296, 28)。假设 embedding_dim=28。
    embedded_image_patches = transformer_preprocessor(
        inputs=image_inputs, patch_size=patch_size,
        embedding_dim=embedding_dim,
        max_weight=max_weight, clip_weight=clip_weight)

    # encoder_outputs 形状为 (batch_size, 1296, 28)。
    encoder_outputs = transformer_encoder(
        embedded_image_patches, use_mish=use_mish,
        num_heads=encoder_num_heads, key_dim=encoder_key_dim,
        dropout=dropout, mlp_quantities=encoder_mlp_quantities,
        repeat_times=encoder_repeat_times,
        max_weight=max_weight, clip_weight=clip_weight, training=training)

    mlp_head_outputs = mlp_head(
        encoder_outputs=encoder_outputs, use_mish=use_mish,
        mlp_head_last_units=mlp_head_last_units,
        dense_layers_quantity=mlp_head_dense_layers_quantity,
        dense_mish_block_repeats=mlp_head_dense_mish_block_repeats,
        dropout=dropout,
        max_weight=max_weight, clip_weight=clip_weight, training=training)

    model = keras.Model(
        inputs=[image_inputs], outputs=mlp_head_outputs,
        name='vision_transformer_detector')

    return model


def transform_predictions(inputs):
    """对模型输出的 MLP Head 进行转换。

    转换方式为：
    MLP Head 的第 1 位是类别置信度，需要将其乘以 79，则数值范围变为 [0, 79]，可以代表 80
    个类别。
    MLP Head 的后 4 位是预测框的位置和坐标，格式为 (x, y, height, width)。需要将它们
    转换为实际的大小：把 x 和 width 乘以图片的宽度，即 MODEL_IMAGE_SIZE[1]，把 y 和
    height 乘以图片的高度，即 MODEL_IMAGE_SIZE[1]。

    Arguments:
        inputs: 一个 tf.float32 类型的张量，张量形状为
            (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。
            最后 1 个维度大小为 6，表示每个预测结果是一个长度为 6 的向量，每一位的数值都
            是 [0, 1] 的范围。各自代表的含义是：
            第 0 位：是类别置信度。
            第 1 位：代表 80 个类别。
            最后 4 位：是预测框的位置和坐标，格式为 (x, y, height, width)，其中 x，y
            是预测框的中心点坐标，height, width 是预测框的高度和宽度。这 4 个数值的范围都
            在 [0, 1] 之间，是一个比例值，表示在原图中的比例大小。
    Returns:
        transformation: 一个 tf.float32 类型的张量，是把 inputs 转换后的结果。
            张量形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。
            最后 1 个维度大小为 6，各自代表的含义是：
            第 0 位：是 [0, 1] 之间的浮点数，代表类别置信度。
            第 1 位：是 [0, 79] 之间的浮点数，代表 80 个类别。
            最后 4 位：是预测框的位置和坐标，格式为 (x, y, height, width)，代表在图片中
            的实际大小，不是比例值。
            其中 x 和 width 是 [0, MODEL_IMAGE_SIZE[1]] 之间的浮点数。
            其中 y 和 height 是 [0, MODEL_IMAGE_SIZE[0]] 之间的浮点数。
    """

    # 尝试把 sigmoid 操作放到模型之外，在转换预测结果部分。
    inputs = keras.activations.sigmoid(inputs)

    # 每一个预测结果的最后 4 位，是一个比例值，不应该超过图片的大小，所以设置比例为 [0, 1]。
    # 另外一个作用是，避免过大的比例值，出现极大的边长，变成 inf，最终得到 NaN。
    clipped_ratio = tf.clip_by_value(inputs[..., -4:], 0, 1)

    inputs = tf.concat([inputs[..., :-4], clipped_ratio], axis=-1)

    # confidence 张量形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 1)。
    confidence = inputs[..., 0: 1]   # 置信度为 [0, 1] 的浮点数，无须进行转换。

    # classification 张量形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 1)。
    # 用 0-79 表示 80 个类别。
    classification = inputs[..., 1: 2] * (Constants.CLASSES.value - 1)

    # 下面是 bbox 的 4 个信息，张量形状均为 (batch_size,
    # MAX_DETECT_OBJECTS_QUANTITY, 1)。
    # 预测框的中心点坐标 x,y,height,width。
    center_x = inputs[..., 2: 3] * Constants.MODEL_IMAGE_SIZE.value[1]
    center_y = inputs[..., 3: 4] * Constants.MODEL_IMAGE_SIZE.value[0]
    bbox_height = inputs[..., 4: 5] * Constants.MODEL_IMAGE_SIZE.value[0]
    bbox_width = inputs[..., 5:] * Constants.MODEL_IMAGE_SIZE.value[1]

    # transformation，张量形状均为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。
    transformation = tf.concat(
        [confidence, classification,
         center_x, center_y, bbox_height, bbox_width], axis=-1)

    return transformation


class CheckModelWeight(keras.callbacks.Callback):
    """在训练过程中，每一个批次训练完之后，检查模型的权重，并给出最大权重，以监视 NaN 的
    发生过程。"""

    def __init__(self, start_epochs, skip_epochs, weight_threshold):
        super(CheckModelWeight, self).__init__()
        # 训练初期权重快速增长，为了避免频繁报告，设置大于 weight_threshold 时才开始报告。
        self.max_weight = weight_threshold
        self.min_weight = -weight_threshold
        self.start_epochs = start_epochs
        self.skip_epochs = skip_epochs

    # noinspection PyUnusedLocal
    def on_epoch_end(self, epoch, logs=None):
        """在训练 start_epochs 个迭代之后，每经过 skip_epochs 个迭代，就对最大权重
        进行一次检查。"""

        if (epoch >= self.start_epochs) and (
                (epoch - self.start_epochs) % self.skip_epochs == 0):

            for layer in self.model.layers:
                if len(layer.weights) > 0:  # 此时该层是有权重的层，需要检查权重。
                    for weight in layer.weights:  # layer.weights 是一个列表。
                        check_inf_nan(inputs=weight, name='weight')

                        if tf.math.reduce_max(weight) > self.max_weight:
                            self.max_weight = tf.math.reduce_max(weight)
                            tf.print(f'\nLargest_weight changed to: '
                                     f'{self.max_weight:.3f}, at epoch {epoch}.'
                                     f'\tWeight shape: {weight.shape}\t\t'
                                     f'Layer name:, {layer.name}')

                        elif tf.math.reduce_min(weight) < self.min_weight:
                            self.min_weight = tf.math.reduce_min(weight)
                            tf.print(f'\nSmallest_weight changed to: '
                                     f'{self.min_weight:.3f}, at epoch {epoch}.'
                                     f'\tWeight shape: {weight.shape}\t\t'
                                     f'Layer name:, {layer.name}')


# 允许学习率衰减的次数。必须使用 tf.Variable。如果使用普通的全局变量，在函数中对其进行
# 修改之后，该全局变量会变为局部变量，导致后续出错。
# 根据 Google Style Guide，全局变量名称之前加上下划线 _。
_allowed_decay_times = tf.Variable(3, trainable=False)


def learning_rate_step_decay(epoch, lr,
                             epochs_first_lr_decay, epochs_second_lr_decay,
                             epochs_third_lr_decay, rate_lr_decay):
    """对学习率进行阶跃衰减。最多允许 3 次衰减。

    Arguments:
        epoch: 一个整数，代表当前的迭代次数。由 Keras 自动输入。
        lr: 一个浮点数，代表当前的学习率。由 Keras 自动输入。
        epochs_first_lr_decay: 一个整数，表示经过 epochs_first_lr_decay 次迭代后，
            进行第 1 次学习率衰减。
        epochs_second_lr_decay: 一个整数，表示经过 epochs_first_lr_decay 和
            epochs_second_lr_decay 次迭代后，进行第 2 次学习率衰减。
        epochs_third_lr_decay: 一个整数，表示经过 epochs_first_lr_decay，
            epochs_second_lr_decay 和 epochs_third_lr_decay 次迭代后，进行第 3 次
            学习率衰减。
        rate_lr_decay: 一个浮点数，是将学习率进行衰减的比率。
    Returns:
        lr: 一个浮点数，是衰减之后的学习率。
    """

    if epoch == epochs_first_lr_decay or (
            epoch == epochs_first_lr_decay + epochs_second_lr_decay) or (
            epoch == epochs_first_lr_decay + epochs_second_lr_decay +
            epochs_third_lr_decay):
        # 在若干个迭代之后，对学习率进行阶跃衰减。同时打印信息，对用户进行提示。
        if _allowed_decay_times.numpy() > 0:
            print('\N{bouquet}'*20 +
                  f'\nChanging the learning rate after epoch {epoch}:\n'
                  f'before change: \t{lr:.2e}')
            lr *= rate_lr_decay
            print(f'after change: \t{lr:.2e}\n' + '\N{bouquet}'*20)
            _allowed_decay_times.assign_sub(1)
    return lr


def check_weights(model_input):
    """检查每个 batch 结束之后，最大权重是否发生变化。主要目的是监视出现极大权重的
    情况。"""

    red_line_weight = 500
    max_weight = 0

    progress_bar = keras.utils.Progbar(
        len(model_input.weights), width=60, verbose=1,
        interval=0.01, stateful_metrics=None, unit_name='step')

    print(f'\nChecking the weights ...')
    # model_input.weights 是一个列表，其中的每一个元素都是一个多维张量。
    for i, weight in enumerate(model_input.weights):
        progress_bar.update(i)

        if max_weight < np.amax(weight):
            max_weight = np.amax(weight)

    if max_weight > red_line_weight:
        print(f'\nAlert! max_weight is: {max_weight:.1f}')
        print('\nVery high weight could lead to a big model output '
              'value, then cause the NaN loss. Please consider:\n'
              '1. use a smaller learning_rate;\n2. reduce the loss value.\n')
    else:
        print(f'\nThe status is OK, max_weight is: {max_weight:.1f}\n')

    return max_weight


def iou_calculator(label_bbox, prediction_bbox):
    """计算预测框和真实框的 IoU 。

    用法说明：使用时，要求输入的 label_bbox, prediction_bbox 形状相同，均为 4D 张量。将
    在两个输入的一一对应的位置上，计算 IoU。
    举例来说，假如两个输入的形状都是 (19, 19, 3, 4)，而标签 label_bbox 只在 (8, 12, 0)
    位置有一个物体框，则 iou_calculator 将会寻找 prediction_bbox 在同样位置
    (8, 12, 0) 的物体框，并计算这两个物体框之间的 IoU。prediction_bbox 中其它位置的物
    体框，并不会和 label_bbox 中 (8, 12, 0) 位置的物体框计算 IoU。
    计算结果的形状为 (19, 19, 3)，并且将在 (8, 12, 0) 位置有一个 IoU 值。

    Arguments:
        label_bbox: 一个 4D 张量，形状为 (input_height, input_width, 3, 4)，代表标
            签中的物体框。
            最后一个维度的 4 个值分别代表物体框的 (center_x, center_y, height_bbox,
            width_bbox)。第 2 个维度的 3 表示有 3 种不同宽高比的物体框。
            该 4 个值必须是实际值，而不是比例值。
        prediction_bbox: 一个 4D 张量，形状为 (input_height, input_width, 3, 4)，
            代表预测结果中的物体框。最后一个维度的 4 个值分别代表物体框的
            (center_x, center_y, height_bbox, width_bbox)。第 2 个维度的 3 表示
            有 3 种不同宽高比的物体框。该 4 个值必须是实际值，而不是比例值。
    Returns:
        iou: 一个 3D 张量，形状为 (input_height, input_width, 3)，代表交并比 IoU。
    """

    # 两个矩形框 a 和 b 相交时，要同时满足的 4 个条件是：
    # left_edge_a < right_edge_b , right_edge_a > left_edge_b
    # top_edge_a < bottom_edge_b , bottom_edge_a > top_edge_b

    # 对每个 bbox，先求出 4 条边。left_edge，right_edge 形状为
    # (input_height, input_width, 3)
    label_left_edge = label_bbox[..., -4] - label_bbox[..., -1] / 2
    label_right_edge = label_bbox[..., -4] + label_bbox[..., -1] / 2

    prediction_left_edge = (prediction_bbox[..., -4] -
                            prediction_bbox[..., -1] / 2)
    prediction_right_edge = (prediction_bbox[..., -4] +
                             prediction_bbox[..., -1] / 2)

    label_top_edge = label_bbox[..., -3] - label_bbox[..., -2] / 2
    label_bottom_edge = label_bbox[..., -3] + label_bbox[..., -2] / 2

    prediction_top_edge = (prediction_bbox[..., -3] -
                           prediction_bbox[..., -2] / 2)
    prediction_bottom_edge = (prediction_bbox[..., -3] +
                              prediction_bbox[..., -2] / 2)

    # left_right_condition 的形状为 (input_height, input_width, 3)
    # 表示 2 个条件：left_edge_a < right_edge_b , right_edge_a > left_edge_b
    left_right_condition = tf.math.logical_and(
        x=(label_left_edge < prediction_right_edge),
        y=(label_right_edge > prediction_left_edge))
    # top_bottom_condition 的形状为 (input_height, input_width, 3)
    # 表示 2 个条件：top_edge_a < bottom_edge_b , bottom_edge_a > top_edge_b
    top_bottom_condition = tf.math.logical_and(
        x=(label_top_edge < prediction_bottom_edge),
        y=(label_bottom_edge > prediction_top_edge))

    # intersection_condition 的形状为
    # (input_height, input_width, 3)，是 4 个条件的总和
    intersection_condition = tf.math.logical_and(x=left_right_condition,
                                                 y=top_bottom_condition)
    # 形状扩展为 (input_height, input_width, 3, 1)
    intersection_condition = tf.expand_dims(intersection_condition, axis=-1)
    # 形状扩展为 (input_height, input_width, 3, 4)
    intersection_condition = tf.repeat(input=intersection_condition,
                                       repeats=4, axis=-1)

    # horizontal_edges, vertical_edges 的形状为
    # (input_height, input_width, 3, 4)
    horizontal_edges = tf.stack(
        values=[label_top_edge, label_bottom_edge,
                prediction_top_edge, prediction_bottom_edge], axis=-1)

    vertical_edges = tf.stack(
        values=[label_left_edge, label_right_edge,
                prediction_left_edge, prediction_right_edge], axis=-1)

    zero_pad_edges = tf.zeros_like(input=horizontal_edges)
    # 下面使用 tf.where，可以使得 horizontal_edges 和 vertical_edges 的形状保持为
    # (input_height, input_width, 3, 4)，并且只保留相交 bbox 的边长值，其它设为 0
    horizontal_edges = tf.where(condition=intersection_condition,
                                x=horizontal_edges, y=zero_pad_edges)
    vertical_edges = tf.where(condition=intersection_condition,
                              x=vertical_edges, y=zero_pad_edges)

    horizontal_edges = tf.sort(values=horizontal_edges, axis=-1)
    vertical_edges = tf.sort(values=vertical_edges, axis=-1)

    # 4 条边按照从小到大的顺序排列后，就可以把第二大的减去第三大的边，得到边长。
    # intersection_height, intersection_width 的形状为
    # (input_height, input_width, 3)
    intersection_height = horizontal_edges[..., -2] - horizontal_edges[..., -3]
    intersection_width = vertical_edges[..., -2] - vertical_edges[..., -3]

    # intersection_area 的形状为 (input_height, input_width, 3)
    intersection_area = intersection_height * intersection_width

    prediction_bbox_width = prediction_bbox[..., -1]
    prediction_bbox_height = prediction_bbox[..., -2]

    # 不能使用混合精度计算。因为 float16 格式下，数值达到 65520 时，就会溢出变为 inf，
    # 从而导致 NaN。而 prediction_bbox_area 的数值是可能达到 320*320 甚至更大的。
    prediction_bbox_area = prediction_bbox_width * prediction_bbox_height

    label_bbox_area = label_bbox[..., -1] * label_bbox[..., -2]

    # union_area 的形状为 (input_height, input_width, 3)
    union_area = prediction_bbox_area + label_bbox_area - intersection_area

    # 为了计算的稳定性，避免出现 nan、inf 的情况，分母可能为 0 时应加上一个极小量 EPSILON
    # iou 的形状为 (input_height, input_width, 3)
    iou = intersection_area / (union_area + Constants.EPSILON.value)

    return iou


def diagonal_calculator(label_bbox, prediction_bbox):
    """计算预测框和真实框最小包络 smallest enclosing box 的对角线长度。

    Arguments:
        label_bbox: 一个 4D 张量，形状为 (height, width, 3, 4)，代表标签中的物体框。
        prediction_bbox: 一个 4D 张量，形状为 (height, width, 3, 4)，代表预测结果中
            的物体框。
    Returns:
        diagonal_length: 一个 3D 张量，形状为 (height, width, 3)，代表预测框和真实框
            最小包络的对角线长度。
    """

    # 要计算最小包络的对角线长度，需要先计算最小包络，4 个步骤如下：
    # 1. 2 个矩形框，共有 4 个水平边缘，按其纵坐标从小到大排列。
    # 2. 把另外 4 个竖直边缘，按横坐标从小到大排列。
    # 3. 将水平边缘的最小值和最大值，加上竖直边缘的最小值和最大值，就形成了最小包络。
    # 4. 根据最小包络的高度和宽度，计算对角线长度。

    # 对每个 bbox，4个参数分别是 (center_x, center_y, height_bbox, width_bbox)

    # 对每个 bbox，先求出 4 条边。left_edge，right_edge 形状为 (height, width, 3)
    label_left_edge = label_bbox[..., -4] - label_bbox[..., -1] / 2
    label_right_edge = label_bbox[..., -4] + label_bbox[..., -1] / 2

    prediction_left_edge = (prediction_bbox[..., -4] -
                            prediction_bbox[..., -1] / 2)
    prediction_right_edge = (prediction_bbox[..., -4] +
                             prediction_bbox[..., -1] / 2)

    label_top_edge = label_bbox[..., -3] - label_bbox[..., -2] / 2
    label_bottom_edge = label_bbox[..., -3] + label_bbox[..., -2] / 2

    prediction_top_edge = (prediction_bbox[..., -3] -
                           prediction_bbox[..., -2] / 2)
    prediction_bottom_edge = (prediction_bbox[..., -3] +
                              prediction_bbox[..., -2] / 2)

    # horizontal_edges, vertical_edges 的形状为 (height, width, 3, 4)
    horizontal_edges = tf.stack(
        values=[label_top_edge, label_bottom_edge,
                prediction_top_edge, prediction_bottom_edge], axis=-1)
    vertical_edges = tf.stack(
        values=[label_left_edge, label_right_edge,
                prediction_left_edge, prediction_right_edge], axis=-1)

    horizontal_edges = tf.sort(values=horizontal_edges, axis=-1)
    vertical_edges = tf.sort(values=vertical_edges, axis=-1)

    # 将水平边缘的最大值减去水平边缘的最小值(这里的最小，是指其坐标值最小)，就得到最小包络的
    # 高度 height_enclosing_box。 height_enclosing_box 的形状为 (height, width, 3)
    height_enclosing_box = horizontal_edges[..., -1] - horizontal_edges[..., 0]

    # 将竖直边缘的最大值减去竖直边缘的最小值(指其坐标值最大)，就得到最小包络的
    # 宽度 width_enclosing_box。 width_enclosing_box 的形状为 (height, width, 3)
    width_enclosing_box = vertical_edges[..., -1] - vertical_edges[..., 0]

    # 将水平边缘和竖直边缘进行 stack 组合，就得到 height_width_enclosing_box。
    # height_width_enclosing_box 的形状为 (height, width, 3, 2)
    height_width_enclosing_box = tf.stack(
        values=[height_enclosing_box, width_enclosing_box], axis=-1)

    # 计算欧氏距离，得到对角线长度 diagonal_length， 其形状为 (height, width, 3)
    diagonal_length = tf.math.reduce_euclidean_norm(
        input_tensor=height_width_enclosing_box, axis=-1)

    return diagonal_length


def ciou_calculator(label_bbox, prediction_bbox, get_diou=None):
    """计算预测框和真实框的 CIOU。

    Arguments:
        label_bbox: 一个 4D 张量，形状为 (height, width, 3, 4)，代表标签中的物体框。
        prediction_bbox: 一个 4D 张量，形状为 (height, width, 3, 4)，代表预测结果中
            的物体框。
        get_diou: 一个布尔值，如果为 True，则返回 diou 的值，在生成 y_true 时会用到。
    Returns:
        loss_ciou: 一个 3D 张量，形状为 (height, width, 3)，代表 CIOU 损失。
    """

    # CIOU loss： loss_ciou = 1 − IoU + r_ciou  https://arxiv.org/abs/1911.08287
    # CIOU 的正则项 regularization: r_ciou = r_diou + α * v

    iou = iou_calculator(label_bbox=label_bbox,
                         prediction_bbox=prediction_bbox)

    # 对每个 bbox，4 个参数分别是 (center_x, center_y, height_bbox, width_bbox)
    label_center = label_bbox[..., : 2]
    prediction_center = prediction_bbox[..., : 2]

    # deltas_x_y 的形状为 (height, width, 3, 2)，代表 2 个 bbox 之间中心点的 x，y差值
    deltas_x_y = label_center - prediction_center
    # 根据论文，用 rho 代表 2 个 bbox 中心点的欧氏距离，形状为 (height, width, 3)
    rho = tf.math.reduce_euclidean_norm(input_tensor=deltas_x_y, axis=-1)

    # c_diagonal_length 的形状为 (height, width, 3)
    c_diagonal_length = diagonal_calculator(label_bbox, prediction_bbox)

    # 为了计算的稳定性，避免出现 nan、inf 的情况，分母可能为 0 时应加上一个极小量 EPSILON
    # 根据论文中的公式 6 得到 r_diou，r_diou 的形状为 (height, width, 3)
    r_diou = tf.math.square(rho / (c_diagonal_length +
                                   Constants.EPSILON.value))

    # 因为论文中的 v 是一个控制宽高比的参数，所以这里将其命名为 v_aspect_ratio
    # 下面根据论文中的公式 9 计算参数 v_aspect_ratio，
    # atan_label_aspect_ratio 的形状为 (height, width, 3)

    atan_label_aspect_ratio = tf.math.atan(
        label_bbox[..., -1] / (label_bbox[..., -2] +
                               Constants.EPSILON.value))
    atan_prediction_aspect_ratio = tf.math.atan(
        prediction_bbox[..., -1] / (prediction_bbox[..., -2] +
                                    Constants.EPSILON.value))

    squared_pi = tf.square(np.pi)
    # 把 squared_pi 转换为混合精度需要的数据类型。
    squared_pi = tf.cast(squared_pi, dtype=tf.float32)

    # v_aspect_ratio 的形状为 (height, width, 3)
    v_aspect_ratio = tf.math.square(
        atan_label_aspect_ratio -
        atan_prediction_aspect_ratio) * 4 / squared_pi

    # 根据论文中的公式 11 得到 alpha， alpha 的形状为 (height, width, 3)
    alpha = v_aspect_ratio / ((1 - iou) +
                              v_aspect_ratio + Constants.EPSILON.value)

    # 根据论文中的公式 8 得到 r_ciou，r_ciou 的形状为 (height, width, 3)
    r_ciou = r_diou + alpha * v_aspect_ratio

    # 根据论文中的公式 10 得到 loss_ciou，loss_ciou 的形状为 (height, width, 3)
    loss_ciou = 1 - iou + r_ciou

    if get_diou:
        diou = iou - r_diou
        return diou

    return loss_ciou


def get_objectness_ignore_mask(y_true, y_pred):
    """根据 IoU，生成对应的 objectness_mask。

    根据 YOLO-V3 论文，当预测结果的物体框和标签的物体框 IoU 大于阈值 0.5 时，则可以
    忽略预测结果物体框预测损失，也就是不计算预测结果的 objectness 损失。

    Arguments:
        y_true: 一个数据类型为 float32 的 5D 张量，代表标签中的物体框。
            计算不同的 head 损失值时，形状不同。计算 p5_true, p4_true, p3_true 时，
            形状分别为 (batch_size, 19, 19, 3, 85)，(batch_size, 38, 38, 3, 85)，
            (batch_size, 76, 76, 3, 85)。
        y_pred: 一个数据类型为 float32 的 5D 张量，代表标签中的物体框。
            计算不同的 head 损失值时，形状不同。计算 p5_prediction, p4_prediction,
            p3_prediction 时，形状分别为 (batch_size, 19, 19, 3, 85)，
            (batch_size, 38, 38, 3, 85)，(batch_size, 76, 76, 3, 85)。

    Returns:
        objectness_mask: 一个形状为 (batch_size, 19, 19, 3) 的 4D 布尔张量（该形状以
            P5 特征层为例），代表可以忽略预测损失 objectness 的物体框。该张量是一个基本全
            为 False 的张量，但是对 IoU 大于阈值 0.5 的物体框，其对应的布尔值为 True。
            此外，对于有物体的预设框，它们的损失值不可以被忽略，所以这些有物体的预设框，它
            们的布尔值会被设置为 False。
    """

    batch_size = y_true.shape[0]

    # 初始的 objectness_mask  形状为 (0, 19, 19, 3)。
    objectness_mask = tf.zeros(shape=(0, *y_true.shape[1: 4]), dtype=tf.bool)

    # 1. 遍历每一张图片和标签。
    for i in range(batch_size):
        # one_label, one_prediction 形状为 (19, 19, 3, 85)。
        one_label = y_true[i]
        one_prediction = y_pred[i]

        # 以 P5 特征层为例，objectness_label 形状为 (19, 19, 3)。
        objectness_label = one_label[..., 0]

        # object_exist_label 形状为 (19, 19, 3)，表示标签中有物体的那些预设框。
        object_exist_label = tf.experimental.numpy.isclose(objectness_label, 1)

        # bboxes_indices_label 形状为 (x, 3)，表示标签中有 x 个预设框，其中有物体。
        bboxes_indices_label = tf.where(object_exist_label)

        # 2. 如果没有标签 bbox，创建全为 False 的布尔张量 objectness_mask_one_image。
        # objectness_mask_one_image 形状为 (19, 19, 3)，是全为 False 的布尔张量。
        objectness_mask_one_image = tf.zeros(
            shape=one_label.shape[: 3], dtype=tf.bool)

        # 3. 如果有标签 bbox，对每一个标签 bbox，计算预测结果中的 IoU。以 IoU 大于阈
        # 值 0.5 为判断条件，得到布尔张量 objectness_mask_one_image。
        if len(bboxes_indices_label) > 0:

            # bboxes_prediction 形状为 (19, 19, 3, 4)，是预测结果中 bboxes 的中心点
            # 坐标和高度宽度信息。
            bboxes_prediction = one_prediction[..., -4:]

            # 3.1 遍历每一个标签 bbox，计算预测结果中的 IoU。
            # indices 形状为 (3,)，是标签中一个 bbox 的索引值。
            for indices in bboxes_indices_label:
                # one_label_info 形状为 (85,)。indices 是标签中一个 bbox 的索引值，
                # 在图模式下，无法将张量转换为元祖，用下面这行代码来代替。
                one_label_info = one_label[indices[0], indices[1], indices[2]]
                # one_label_bbox 形状为 (4,)。
                one_label_bbox = one_label_info[-4:]

                # bbox_label 形状为 (19, 19, 3, 4)，张量中每一个长度为 (4,)的向量，
                # 都是 one_bbox_label 的中心点坐标和高度宽度信息。
                bbox_label = tf.ones_like(bboxes_prediction) * one_label_bbox

                # iou_result 形状为 (19, 19, 3)，是预测结果 bboxes 的 IoU。
                iou_result = iou_calculator(label_bbox=bbox_label,
                                            prediction_bbox=bboxes_prediction)

                # objectness_mask_one_label_bbox 形状为 (19, 19, 3)。
                objectness_mask_one_label_bbox = (iou_result > 0.5)

                # 3.2 对每个标签 bbox 得到的 objectness_mask_one_label_bbox，使用
                # 逻辑或操作，得到最终的 objectness_mask_one_image。
                # objectness_mask_one_image 形状为 (19, 19, 3)。
                objectness_mask_one_image = tf.math.logical_or(
                    objectness_mask_one_image, objectness_mask_one_label_bbox)

            # 3.3 把所有标签 bboxes 遍历完成之后，对于标签中有物体框的位置，还要把
            # objectness_mask_one_image 的对应位置设为 False。即这些有物体的预设框
            # 损失值不可以被忽略，必须计算二元交叉熵损失。

            # objectness_mask_one_image 形状为 (19, 19, 3)。
            objectness_mask_one_image = tf.where(
                condition=object_exist_label,
                x=False, y=objectness_mask_one_image)

        # 4. 将 batch_size 个 objectness_mask_one_image 进行 concatenate，得到最终
        # 的 objectness_mask。
        # objectness_mask_one_image 形状为 (1, 19, 19, 3)。
        objectness_mask_one_image = objectness_mask_one_image[tf.newaxis, ...]

        # objectness_mask 最终的形状为 (batch_size, 19, 19, 3)。
        objectness_mask = tf.concat(
            values=[objectness_mask, objectness_mask_one_image], axis=0)

    return objectness_mask


def my_custom_loss(y_true, y_pred, focal_binary_loss=True,
                   coefficient=4, exponent=2,
                   weight_classification=0.0074, weight_ciou=10,
                   use_transform_predictions=True):
    """Vision Transformer Detector 的自定义损失函数。

    该损失函数包括 3 部分：
    1. 对每个物体框，判断框内是否有物体，使用 focal loss 形式的二元交叉熵损失。
    2. 分类损失使用指数函数的形式，即 loss = (coefficient * error) ** exponent。其中
        error 为预测类别和标签类别的差值，coefficient 是一个常数，exponent 是一个偶数。
        例如可以使用 loss = (4*error) ** 4 的形式（可以称之为 4x4 Off-road 损失函数）。
    3. 物体框的位置使用 CIOU 损失。

    Arguments:
        y_true: 一个 tf.float32 类型的张量，代表标签中的物体框。
            张量形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。
            最后 1 个维度大小为 6。这 6 个数值各自代表的含义是：
            第 0 位：是 0 或者 1，代表类别置信度。0 表示物体框内没有物体，1 表示物
            体框内有物体。
            第 1 位：是 [0, 79] 之间的整数，代表 80 个类别。如果物体框内没有物体，则
            该位数值等于 -8。
            最后 4 位：是物体框的位置和坐标，格式为 (x, y, height, width)，代表在
            图片中的实际大小，不是比例值。如果物体框内没有物体，则此 4 位数值等于 -8。
            其中 x 和 width 是 [0, MODEL_IMAGE_SIZE[1]] 之间的浮点数。
            其中 y 和 height 是 [0, MODEL_IMAGE_SIZE[0]] 之间的浮点数。
        y_pred: 一个数据类型为 float32 的 3D 张量，代表预测结果中的物体框。张量形状为
            (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。其中每一个长度为 6 的向
            量，其含义和 y_true 中的相同。
        focal_binary_loss: 一个布尔值，如果为 True，则使用 focal loss 形式的交叉熵。
        coefficient: 一个浮点数，是分类损失函数中的系数。
        exponent: 一个浮点数，是分类损失函数中的指数。
        weight_classification: 一个浮点数，是分类损失值的权重系数。
        weight_ciou: 一个浮点数，是探测框损失值的权重系数。
        use_transform_predictions: 一个布尔值，如果为 True，将使用函数
            transform_predictions 对预测结果进行转换。

    Returns:
        total_loss: 一个浮点数，代表该批次的平均损失值，等于总的损失值除以批次大小。
    """
    # y_pred = check_inf_nan(inputs=y_pred, replace_nan=0.,
    #                        name='Debug 1, before y_pred')
    if use_transform_predictions:
        y_pred = transform_predictions(inputs=y_pred)
    # check_inf_nan(inputs=y_pred, name='Debug 2, after y_pred')

    # 使用二元交叉熵来计算 objectness 损失。
    if focal_binary_loss:
        # 尝试使用 focal loss。
        binary_crossentropy_logits_false = keras.losses.BinaryFocalCrossentropy(
            from_logits=False, label_smoothing=0,
            gamma=2.0,
            reduction=keras.losses.Reduction.NONE)
    else:
        # 使用 BinaryCrossentropy，后续根据需要使用 label_smoothing。
        binary_crossentropy_logits_false = keras.losses.BinaryCrossentropy(
            from_logits=False, label_smoothing=0,
            reduction=keras.losses.Reduction.NONE)

    # label_objectness 的形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 1)。
    label_objectness = y_true[..., 0: 1]
    prediction_objectness = y_pred[..., 0: 1]

    # 经过 binary_crossentropy 计算后，loss_objectness 的形状为
    # (batch_size, MAX_DETECT_OBJECTS_QUANTITY)，会去掉最后一个维度。
    loss_objectness = binary_crossentropy_logits_false(
        label_objectness, prediction_objectness)

    # 不需要使用 ignore_mask，因为 YOLOv4-CSP 有许多预设框 anchor boxes，存在多个预设
    # 框预测同一个物体的问题。而 ViT 探测器因为是 anchor free 的算法，所以没有这个问题。
    # loss_objectness_mean 是一个标量。
    loss_objectness_mean = tf.reduce_mean(loss_objectness)

    # object_exist_tensor 是一个布尔张量，形状为 (batch_size,
    # MAX_DETECT_OBJECTS_QUANTITY)，
    # 用于判断预设框内是否有物体。如果有物体，则对应的位置为 True。
    # 因为浮点数没有精度，不能直接比较是否相等，应该用 isclose 函数进行比较。
    object_exist_tensor = tf.experimental.numpy.isclose(y_true[..., 0], 1.0)

    # 计算损失的均值，先要统计有多少个样本参与了计算损失值。 existing_objects 是一个张量，
    # 包含若干个元祖，代表了 object_exist_tensor 中为 True 的元素的索引值。
    existing_objects = tf.where(object_exist_tensor)

    # object_exist_boxes_quantity 是一个整数型张量，表示标签中正样本的数量。
    object_exist_boxes_quantity = len(existing_objects)
    # 只在有物体的探测框，才计算分类损失和 CIOU 损失，否则会产生 NaN 损失。
    if object_exist_boxes_quantity > 0:

        # label_classification 的形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY)。
        label_classification = y_true[..., 1]
        # label_classification 的形状为 (object_exist_boxes_quantity)。
        label_classification = label_classification[object_exist_tensor]

        # prediction_classification 的形状为
        # (batch_size, MAX_DETECT_OBJECTS_QUANTITY)。
        prediction_classification = y_pred[..., 1]
        # prediction_classification 的形状为 (object_exist_boxes_quantity)。
        prediction_classification = prediction_classification[
            object_exist_tensor]

        # classification_error 的形状为 (object_exist_boxes_quantity)。
        classification_error = tf.abs(prediction_classification -
                                      label_classification)

        # loss_classification 的形状为 (object_exist_boxes_quantity)。
        loss_classification = (coefficient * classification_error) ** exponent

        loss_classification_mean = tf.reduce_mean(loss_classification)

        # 下面计算物体框的损失。
        # label_bbox 张量的形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 4)。
        label_bbox = y_true[..., -4:]
        # label_bbox 的形状为 (object_exist_boxes_quantity, 4)。
        label_bbox = label_bbox[object_exist_tensor]

        # prediction_bbox 张量的形状为
        # (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 4)。
        prediction_bbox = y_pred[..., -4:]
        # prediction_bbox 的形状为 (object_exist_boxes_quantity, 4)。
        prediction_bbox = prediction_bbox[object_exist_tensor]

        # loss_ciou 形状为 (object_exist_boxes_quantity,)，注意 ciou_calculator
        # 函数会消去输入张量的最后一个维度。
        loss_ciou = ciou_calculator(label_bbox=label_bbox,
                                    prediction_bbox=prediction_bbox)

        loss_ciou_mean = tf.reduce_mean(loss_ciou)

    else:
        loss_classification_mean = 0.
        loss_ciou_mean = 0.

    # check_inf_nan(inputs=loss_objectness_mean, name='Debug 3, at total_loss')
    # check_inf_nan(inputs=loss_classification_mean,
    # name='Debug 4, at total_loss')
    # check_inf_nan(inputs=loss_ciou_mean, name='Debug 5, at total_loss')

    total_loss = (loss_objectness_mean +
                  loss_classification_mean * weight_classification +
                  loss_ciou_mean * weight_ciou)
    # 如果损失为 NaN，则直接把它改为 0.0001，避免后续导致梯度和 weight 全都变为 NaN。
    # total_loss = check_inf_nan(inputs=total_loss, replace_nan=0.0001,
    #                            name='Debug 6, total_loss')

    return total_loss


class MeanAveragePrecision(tf.keras.metrics.Metric):
    """计算 COCO 的 AP 指标。

    使用说明：COCO 的 AP 指标，是 10 个 IoU 阈值下，80 个类别 AP 的平均值，即 mean
    average precision。为了和单个类别的 AP 进行区分，这里使用 mAP 来代表 AP 的平均值。

    受内存大小的限制，对每一个类别，只使用最近 LATEST_RELATED_IMAGES 张相关图片计算其
    AP(COCO 实际是使用所有相关图片)。
    相关图片是指该图片的标签或是预测结果的正样本中，包含了该类别。对每个类别的每张图片，
    只保留 BBOXES_PER_IMAGE 个 bboxes 来计算 AP（COCO 实际是最多使用 100 个 bboxes）。
    """

    def __init__(self, name='AP', **kwargs):
        super().__init__(name=name, **kwargs)

        # latest_positive_bboxes: 一个 tf.Variable 张量，用于存放最近的
        # LATEST_RELATED_IMAGES 张相关图片，且每张图片只保留 BBOXES_PER_IMAGE 个
        # positive bboxes，每个 bboxes 有 2 个数值，分别是类别置信度，以及 IoU 值。
        self.latest_positive_bboxes = tf.Variable(
            tf.zeros(
                shape=(Constants.CLASSES.value,
                       Constants.LATEST_RELATED_IMAGES.value,
                       Constants.BBOXES_PER_IMAGE.value, 2)),
            trainable=False, name='latest_positive_bboxes')

        # labels_quantity_per_image: 一个形状为 (CLASSES, BBOXES_PER_IMAGE) 的
        # 整数型张量，表示每张图片中，该类别的标签 bboxes 数量。
        self.labels_quantity_per_image = tf.Variable(
            tf.zeros(shape=(Constants.CLASSES.value,
                            Constants.LATEST_RELATED_IMAGES.value)),
            trainable=False, name='labels_quantity_per_image')

        # showed_up_classes：一个形状为 (CLASSES, ) 的布尔张量，用于记录所有出现过的
        # 类别。每批次数据中，都会出现不同的类别，计算指标时，只使用出现过的类别进行计算。
        self.showed_up_classes = tf.Variable(
            tf.zeros(shape=(Constants.CLASSES.value,), dtype=tf.bool),
            trainable=False, name='showed_up_classes')

        # reset_state 必须放在 3 个属性的后面，因为要先创建这些属性，再对它们进行清零。
        self.reset_state()

    # noinspection PyUnusedLocal, PyMethodMayBeStatic
    def update_state(self, y_true, y_pred, sample_weight=None,
                     use_transform_predictions=True):
        """根据每个 batch 的计算结果，区分 4 种情况，更新状态 state。

        Arguments:
            y_true: 一个浮点类型张量，形状为
                (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。是每个批次数据的标签。
                最后 1 个维度大小为 6。这 6 个数值各自代表的含义是：
                第 0 位：是 0 或者 1，代表类别置信度。0 表示物体框内没有物体，1 表示物
                体框内有物体。
                第 1 位：是 [0, 79] 之间的整数，代表 80 个类别。如果物体框内没有物体，则
                该位数值等于 -8。
                最后 4 位：是物体框的位置和坐标，格式为 (x, y, height, width)，代表在
                图片中的实际大小，不是比例值。如果物体框内没有物体，则此 4 位数值等于 -8。
                其中 x 和 width 是 [0, MODEL_IMAGE_SIZE[1]] 之间的浮点数。
                其中 y 和 height 是 [0, MODEL_IMAGE_SIZE[0]] 之间的浮点数。
            y_pred: 一个数据类型为 float32 的 3D 张量，代表预测结果中的物体框。张量
                形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。其中每一个
                长度为 6 的向量，其含义和 y_true 中的相同。
            sample_weight: update_state 方法的必备参数，即使不使用该参数，也必须在此
                进行定义，否则程序会报错(TF 2.8 版本时会报错，后续版本未知)。
            use_transform_predictions: 一个布尔值，如果为 True，将使用函数
                transform_predictions 对预测结果进行转换。
                当使用测试盒 testcase 时，在每个单元测试中设置
                transform_predictions=False，因为测试盒的 y_pred 是已经转换完成后
                的结果，不需要用 transform_predictions 再次转换。
        """

        # 先将模型输出进行转换。y_pred 形状为
        # (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。
        if use_transform_predictions:
            y_pred = transform_predictions(inputs=y_pred)

        # 先更新第一个状态量 showed_up_classes，更新该状态量不需要逐个图片处理。
        # 1. 先从标签中提取所有出现过的类别。
        # showed_up_categories_label 形状为
        # (batch_size, MAX_DETECT_OBJECTS_QUANTITY)。
        categories_label = y_true[..., 1]

        # showed_up_categories_label 形状为 (x,)，里面存放的是出现过的类别编号，表示
        # 有 x 个类别出现在了这批标签中。因为在标签中，对于没有物体的框，其类别编号为 -8，
        # 所以可以用类别编号是否 ≥ 0 来判断该物体框内是否有物体。
        showed_up_categories_label = categories_label[categories_label >= 0]

        # showed_up_categories_label 形状为 (1, x)。
        showed_up_categories_label = showed_up_categories_label[tf.newaxis, :]

        # 2. 从预测结果中提取所有出现过的类别。
        # objectness_pred 形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY)。
        objectness_pred = y_pred[..., 0]

        # classification_pred 形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY)，
        # 表示的是预测结果中的分类部分，其中数值为 [0, 79] 之间的浮点数。
        classification_pred = y_pred[..., 1]

        # categories_pred 形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY)，
        # 表示的是类别 ID，其中数值为 [0, 79] 之间的整数。
        categories_pred = tf.round(classification_pred)

        # classification_error 形状为 (batch_size, MAX_DETECT_OBJECTS_QUANTITY)，
        # 表示的是预测结果中的类别误差，即预测的浮点数和该类别的整数之间的差值。
        classification_error = tf.abs(classification_pred - categories_pred)

        # classification_confidence_pred 形状为
        # (batch_size, MAX_DETECT_OBJECTS_QUANTITY)，表示的是类别的置信度。这里用
        #  0.5 作为阈值，是因为当预测值和类别 id 相差 0.5 以上时，将被认为是另外一个类别。
        classification_confidence_pred = (0.5 - classification_error) / 0.5

        # showed_up_categories_index_pred 形状为
        # (batch_size, MAX_DETECT_OBJECTS_QUANTITY)，是一个布尔张量。和 y_true
        # 不同的地方在于，它需要大于 2 个置信度阈值，才认为是做出了预测，得出正确的布尔张量。
        showed_up_categories_index_pred = tf.logical_and(
            x=(objectness_pred > Constants.OBJECTNESS_THRESHOLD.value),
            y=(classification_confidence_pred >
               Constants.CLASSIFICATION_CONFIDENCE_THRESHOLD.value))

        # showed_up_categories_pred 形状为 (y,)，里面存放的是出现过的类别编号，表示
        # 有 y 个类别出现在了这批预测结果中。
        showed_up_categories_pred = categories_pred[
            showed_up_categories_index_pred]

        # showed_up_categories_pred 形状为 (1, y)。
        showed_up_categories_pred = showed_up_categories_pred[tf.newaxis, :]

        # 改为 int 类型之后，才能参与下面的 union 计算。
        showed_up_categories_label = tf.cast(showed_up_categories_label,
                                             dtype=tf.int32)
        showed_up_categories_pred = tf.cast(showed_up_categories_pred,
                                            dtype=tf.int32)
        # showed_up_categories 形状为 (z,)，是一个 sparse tensor。对出现过的类别求
        # 并集，数量从 x,y 变为 z。
        showed_up_categories = tf.sets.union(showed_up_categories_pred,
                                             showed_up_categories_label)

        # 将 showed_up_categories 从 sparse tensor 转化为 tf.tensor。
        showed_up_categories = showed_up_categories.values

        # 更新状态量 showed_up_classes。
        # 遍历该 batch 中的每一个类别，如果该类别是第一次出现，则需要将其记录下来。
        for showed_up_category in showed_up_categories:
            if not self.showed_up_classes[showed_up_category]:
                self.showed_up_classes[showed_up_category].assign(True)

        # 下面更新另外 2 个状态量 latest_positive_bboxes 和
        # labels_quantity_per_image，需要逐个图片处理。
        batch_size = y_true.shape[0]

        # 步骤 1，遍历每一张图片预测结果及其对应的标签。
        for sample in range(batch_size):

            # one_label 形状为 (MAX_DETECT_OBJECTS_QUANTITY, 6)。每一个长度为 6 的
            # 向量，记录的是物体框信息。对于没有物体的物体框，其类别 id 和物体框信息共
            #  5 位，都是数值 -8.
            one_label = y_true[sample]
            # one_pred 形状为 (MAX_DETECT_OBJECTS_QUANTITY, 6)。
            one_pred = y_pred[sample]

            # 步骤 2.1，对于标签，构造张量 categories_one_label。
            # categories_one_label 形状为 (MAX_DETECT_OBJECTS_QUANTITY,)，对于
            # 有物体的物体框，记录的是其类别编号；而在没有物体的物体框，其数值为 -8。
            categories_one_label = one_label[..., 1]

            # 步骤 2.2，对于预测结果，构造 3 个张量：positives_index_one_pred，
            # positives_one_pred 和 categories_one_pred。

            # objectness_one_pred 形状为 (MAX_DETECT_OBJECTS_QUANTITY,)。
            objectness_one_pred = one_pred[..., 0]

            # classification_one_pred 形状为 (MAX_DETECT_OBJECTS_QUANTITY,)，表
            # 示的是预测结果中的分类部分，其中数值为 [0, 79] 之间的浮点数。
            classification_one_pred = one_pred[..., 1]

            # categories_one_pred 形状为 (MAX_DETECT_OBJECTS_QUANTITY,)，表示的
            # 是类别 ID，其中数值为 [0, 79] 之间的整数。
            categories_one_pred = tf.round(classification_one_pred)

            # classification_error_one_pred 形状为
            # (MAX_DETECT_OBJECTS_QUANTITY,)，表示的是预测结果中的类别误差，即预测的
            # 浮点数和该类别的整数之间的差值。
            classification_error_one_pred = tf.abs(
                classification_one_pred - categories_one_pred)

            # classification_confidence_one_pred 形状为
            # (MAX_DETECT_OBJECTS_QUANTITY,)，表示的是类别的置信度。这里用 0.5 作
            # 为阈值，是因为当预测值和类别 ID 相差 0.5 以上时，将被认为是另外一个类别。
            classification_confidence_one_pred = (
                0.5 - classification_error_one_pred) / 0.5

            # positives_index_one_pred 形状为
            # (MAX_DETECT_OBJECTS_QUANTITY,)，是一个布尔张量。
            positives_index_one_pred = tf.logical_and(
                x=(objectness_one_pred > Constants.OBJECTNESS_THRESHOLD.value),
                y=(classification_confidence_one_pred >
                   Constants.CLASSIFICATION_CONFIDENCE_THRESHOLD.value))

            # positives_one_pred 形状为 (MAX_DETECT_OBJECTS_QUANTITY, 6)，是预测
            # 结果正样本的信息，在负样本的位置，其数值为 -8。
            positives_one_pred = tf.where(
                condition=positives_index_one_pred[..., tf.newaxis],
                x=one_pred, y=-8.)

            # positives_category_one_pred 形状为 (MAX_DETECT_OBJECTS_QUANTITY,)，
            # 是预测结果正样本的类别编号，在负样本的位置，其数值为 -8。
            positives_category_one_pred = tf.where(
                condition=positives_index_one_pred,
                x=categories_one_pred, y=-8.)

            # 步骤 3，遍历所有 80 个类别，更新另外 2 个状态值。
            # 对于每一个类别，可能会在 y_true, y_pred 中出现，也可能不出现。组合起来
            # 有 4 种情况，需要对这 4 种情况进行区分，更新状态值。
            for category in range(Constants.CLASSES.value):

                # category_bool_label 和 category_bool_pred 形状都为
                # (MAX_DETECT_OBJECTS_QUANTITY,)，所有属于当前类别的 bboxes，其布尔
                # 值为 True。这也是把 categories_one_label，
                # positives_category_one_pred 的负样本设置为 -8 的原因，
                # 避免和 category 0 发生混淆。
                category_bool_label = tf.experimental.numpy.isclose(
                    categories_one_label, category)
                category_bool_pred = tf.experimental.numpy.isclose(
                    positives_category_one_pred, category)

                # category_bool_any_label 和 category_bool_any_pred 是单个布尔值，
                # 用于判断 4 种情况。
                category_bool_any_label = tf.reduce_any(category_bool_label)
                category_bool_any_pred = tf.reduce_any(category_bool_pred)

                # 下面要分 4 种情况，更新状态量。
                # 情况 a ：标签和预测结果中，都没有该类别。无须更新状态。

                # 情况 b ：预测结果中没有该类别，但是标签中有该类别。
                # 对于预测结果，要提取置信度和 IoU，且置信度和 IoU 都为 0。
                # 对于标签，则提取该类别的标签数量即可。
                # scenario_b 是单个布尔值。
                scenario_b = tf.logical_and((~category_bool_any_pred),
                                            category_bool_any_label)

                # 情况 c ：预测结果中有该类别，标签没有该类别。
                # 对于预测结果，要提取置信度，而因为没有标签，IoU 为 0。
                # 对于标签，提取该类别的标签数量为 0 即可。
                # scenario_c 是单个布尔值。
                scenario_c = tf.logical_and(category_bool_any_pred,
                                            (~category_bool_any_label))

                # 情况 d ：预测结果和标签中都有该类别，此时要计算 IoU，再提取预测结果的
                # 置信度和 IoU，标签中则要提取标签数量。scenario_d 是单个布尔值。
                scenario_d = tf.logical_and(category_bool_any_pred,
                                            category_bool_any_label)

                # 只有在情况 b, c, d 时，才需要更新状态，所以先要判断是否处在情况
                # b, c, d 下。under_scenarios_bcd 是单个布尔值。
                under_scenarios_bc = tf.logical_or(scenario_b, scenario_c)
                under_scenarios_bcd = tf.logical_or(under_scenarios_bc,
                                                    scenario_d)

                # 在情况 b, c, d 时，更新状态量。
                if under_scenarios_bcd:
                    # 更新第二个状态量 labels_quantity_per_image，其形状为
                    # (Constants.CLASSES.value, LATEST_RELATED_IMAGES)。
                    # one_image_category_labels_quantity 是一个整数，表示在一张图
                    # 片中，属于当前类别的标签 bboxes 数量。
                    one_image_category_labels_quantity = tf.where(
                        category_bool_label).shape[0]

                    # 如果某个类别没有在标签中出现，标签数量会是个 None，需要改为 0 。
                    if one_image_category_labels_quantity is None:
                        one_image_category_labels_quantity = 0

                    # 先把 labels_quantity_per_image 整体后移一位。
                    self.labels_quantity_per_image[category, 1:].assign(
                        self.labels_quantity_per_image[category, :-1])

                    # 把最近一个标签数量更新到 labels_quantity_per_image 的第 0 位。
                    self.labels_quantity_per_image[category, 0].assign(
                        one_image_category_labels_quantity)

                    # 最后更新第三个状态量 latest_positive_bboxes，形状为
                    # (CLASSES, LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)。
                    # 需要对 3 种情况 b,c,d 分别进行更新。

                    # 情况 b ：预测结果中没有该类别，但是标签中有该类别。
                    # 对于预测结果，要提取置信度和 IoU，且置信度和 IoU 都为 0。
                    if scenario_b:

                        # one_image_positive_bboxes 形状为 (BBOXES_PER_IMAGE, 2)。
                        one_image_positive_bboxes = tf.zeros(
                            shape=(Constants.BBOXES_PER_IMAGE.value, 2))

                    # 情况 c：预测结果中有该类别，标签没有该类别。
                    # 对于预测结果的状态，要提取置信度，而因为没有标签，IoU 为 0。
                    elif scenario_c:
                        # scenario_c_positives_pred 形状为(scenario_c_bboxes, 6)。
                        scenario_c_positives_pred = positives_one_pred[
                            category_bool_pred]

                        # scenario_c_categories_pred 形状为 (scenario_c_bboxes,)。
                        scenario_c_categories_pred = scenario_c_positives_pred[
                                                     :, 1]
                        # scenario_c_categories 形状为 (scenario_c_bboxes,)。
                        scenario_c_categories = tf.round(
                            scenario_c_categories_pred)
                        # scenario_c_categories_error 形状为(scenario_c_bboxes,)。
                        scenario_c_categories_error = tf.abs(
                            scenario_c_categories_pred - scenario_c_categories)
                        # scenario_c_class_confidence_pred 形状为
                        # (scenario_c_bboxes,)。
                        scenario_c_class_confidence_pred = (
                            0.5 - scenario_c_categories_error) / 0.5

                        scenario_c_bboxes = (
                            scenario_c_class_confidence_pred.shape[0])

                        if scenario_c_bboxes is None:
                            scenario_c_bboxes = 0

                        # 如果 scenario_c_bboxes 数量少于规定的数量，则进行补零。
                        if scenario_c_bboxes < Constants.BBOXES_PER_IMAGE.value:
                            # scenario_c_paddings 形状为 (1, 2)。
                            scenario_c_paddings = tf.constant(
                                (0, (Constants.BBOXES_PER_IMAGE.value -
                                     scenario_c_bboxes)), shape=(1, 2))

                            # scenario_c_confidence_pred 形状为
                            # (BBOXES_PER_IMAGE,)。
                            scenario_c_confidence_pred = tf.pad(
                                tensor=scenario_c_class_confidence_pred,
                                paddings=scenario_c_paddings,
                                mode='CONSTANT', constant_values=0)

                        # 如果 scenario_c_bboxes 数量大于等于规定的数量，则应该先按
                        # 类别置信度从大到小的顺序进行排序，然后保留规定的数量 bboxes。
                        else:
                            # scenario_c_sorted_pred 形状为
                            # (BBOXES_PER_IMAGE,)。
                            scenario_c_sorted_pred = tf.sort(
                                scenario_c_class_confidence_pred,
                                direction='DESCENDING')

                            # scenario_c_confidence_pred 形状为
                            # (BBOXES_PER_IMAGE,)。
                            scenario_c_confidence_pred = (
                                scenario_c_sorted_pred[
                                    : Constants.BBOXES_PER_IMAGE.value])

                        # scenario_c_ious_pred 形状为 (BBOXES_PER_IMAGE,)。
                        scenario_c_ious_pred = tf.zeros_like(
                            scenario_c_confidence_pred)

                        # one_image_positive_bboxes 形状为 (BBOXES_PER_IMAGE, 2)。
                        one_image_positive_bboxes = tf.stack(
                            values=[scenario_c_confidence_pred,
                                    scenario_c_ious_pred], axis=1)

                    # 情况 d ：预测结果和标签中都有该类别，此时要计算 IoU，再提取预测结果
                    # 的置信度和 IoU，标签中则要提取标签数量。scenario_d 是单个布尔值。
                    else:
                        # 1. bboxes_iou_pred 形状为 (MAX_DETECT_OBJECTS_QUANTITY,
                        # 4)，是预测结果正样本的信息，在负样本的位置，其数值为 -8。
                        # 注意必须使用 tf.where 和 positives_one_pred，过滤出当
                        # 前的类别，并保持形状不变。
                        bboxes_iou_pred = tf.where(
                            condition=category_bool_pred[..., tf.newaxis],
                            x=positives_one_pred[..., -4:], y=-8.)

                        # 2. 构造 bboxes_category_label， 形状为
                        # (scenario_d_bboxes_label, 4)。
                        bboxes_category_label = one_label[..., -4:][
                            category_bool_label]

                        # bboxes_area_label 形状为 (scenario_d_bboxes_label,)，
                        # 是当前类别中，各个 bbox 的面积。
                        bboxes_area_label = (bboxes_category_label[:, -1] *
                                             bboxes_category_label[:, -2])

                        # 把标签的 bboxes 按照面积从小到大排序。
                        # sort_by_area 形状为 (scenario_d_bboxes_label,)
                        sort_by_area = tf.argsort(values=bboxes_area_label,
                                                  axis=0, direction='ASCENDING')

                        # 3. 构造 sorted_bboxes_label， 形状为
                        # (scenario_d_bboxes_label, 4)。
                        sorted_bboxes_label = tf.gather(
                            params=bboxes_category_label,
                            indices=sort_by_area, axis=0)

                        # 4. 用 one_image_positive_bboxes 记录下新预测的且命中标签的
                        # bboxes，直接设置其为空，后续用 concat 方式添加新的 bboxes。
                        one_image_positive_bboxes = tf.zeros(
                            shape=(Constants.BBOXES_PER_IMAGE.value, 2))

                        # 用 new_bboxes_quantity 作为标识 flag，每向
                        # one_image_positive_bboxes 增加一个 bbox 信息，则变大 1.
                        new_bboxes_quantity = 0

                        # 5. 遍历 sorted_bboxes_label。
                        for bbox_info in sorted_bboxes_label:

                            # carried_over_shape 形状为
                            # (MAX_DETECT_OBJECTS_QUANTITY, 4)。
                            carried_over_shape = tf.ones_like(bboxes_iou_pred)

                            # 5.1 bbox_iou_label 形状为
                            # (MAX_DETECT_OBJECTS_QUANTITY, 4)。
                            bbox_iou_label = carried_over_shape * bbox_info

                            # 5.2 ious_category 形状为
                            # (MAX_DETECT_OBJECTS_QUANTITY,)。
                            # 因为预测结果中的负样本位置数值是 -8，所以不会和标签中的
                            # bboxes 相交，其对应 IoU 等于 0。
                            ious_category = iou_calculator(
                                label_bbox=bbox_iou_label,
                                prediction_bbox=bboxes_iou_pred)

                            # max_iou_category 是一个标量，表示当前类别所有 bboxes，
                            # 计算得到的最大 IoU。
                            max_iou_category = tf.reduce_max(ious_category)

                            # 5.3 当最大 IoU 大于 0.5 时，则认为预测的 bbox 命中了该
                            # 标签，需要把置信度和 IoU 记录到 category_new_bboxes 中。
                            if max_iou_category > 0.5:
                                new_bboxes_quantity += 1

                                # max_iou_position 是一个布尔张量，仅最大 IoU 位置
                                # 为 True。形状为 (MAX_DETECT_OBJECTS_QUANTITY,)。
                                max_iou_position = (
                                    tf.experimental.numpy.isclose(
                                        ious_category, max_iou_category))

                                # max_iou_bbox_pred 形状为 (1, 6)，是预测结果中
                                # IoU 最大的那个 bbox。
                                max_iou_bbox_pred = positives_one_pred[
                                    max_iou_position]

                                # max_iou_bbox_classification 是预测的分类结果，
                                # 是一个浮点数。
                                max_iou_bbox_classification = (
                                    max_iou_bbox_pred[0, 1])

                                # max_iou_bbox_category 是取整后的类别 ID，
                                # 是一个形式为 x.0 的整数。
                                max_iou_bbox_category = tf.round(
                                    max_iou_bbox_classification)

                                max_iou_bbox_class_error = tf.abs(
                                    max_iou_bbox_classification -
                                    max_iou_bbox_category)

                                # 计算置信度时，必须以 0.5 为阈值。
                                max_iou_bbox_class_confidence = (
                                    0.5 - max_iou_bbox_class_error) / 0.5

                                # new_bbox 是一个元祖，包含类别置信度和 IoU。
                                new_bbox = (max_iou_bbox_class_confidence,
                                            max_iou_category)

                                # new_bbox 形状为 (1, 2)。
                                new_bbox = tf.ones(shape=(1, 2)) * new_bbox

                                # 记录这个命中标签的 bbox 信息。append_new_bboxes
                                # 形状为 (BBOXES_PER_IMAGE + 1, 2)。
                                append_new_bboxes = tf.concat(
                                    values=[one_image_positive_bboxes,
                                            new_bbox], axis=0)

                                # 5.3.1 记录到 one_image_positive_bboxes， 形状为
                                # (BBOXES_PER_IMAGE, 2)。
                                one_image_positive_bboxes = (
                                    append_new_bboxes[
                                        -Constants.BBOXES_PER_IMAGE.value:])

                                # 5.3.2 需要将该 bbox 从 bboxes_iou_pred
                                # 中移除，再进行后续的 IoU 计算。remove_max_iou_bbox
                                # 形状为 (MAX_DETECT_OBJECTS_QUANTITY, 1)，在最
                                # 大 IoU 的位置为 True，其它为 False。
                                remove_max_iou_bbox = max_iou_position[
                                    ..., tf.newaxis]

                                # bboxes_iou_pred 形状为
                                # (MAX_DETECT_OBJECTS_QUANTITY, 4)。
                                # 把被去除的 bbox 替换为 -8。
                                bboxes_iou_pred = tf.where(
                                    condition=remove_max_iou_bbox,
                                    x=-8., y=bboxes_iou_pred)

                            # 5.4 当记录的数量达到 BBOXES_PER_IMAGE 之后，跳出
                            # 遍历 sorted_bboxes_label 的循环，停止记录新的 bboxes。
                            if (new_bboxes_quantity ==
                                    Constants.BBOXES_PER_IMAGE.value):
                                break

                        # 6. 遍历 sorted_bboxes_label 完成之后，处理
                        # bboxes_iou_pred 中剩余的 bboxes。

                        # bboxes_iou_pred 形状为 (MAX_DETECT_OBJECTS_QUANTITY,
                        # 4)，是预测结果正样本的信息；而在负样本的位置，其数值为 -8。
                        # left_bboxes_bool 形状为 (MAX_DETECT_OBJECTS_QUANTITY,)，
                        # 是一个布尔张量，剩余 bboxes 位置为 True，其它为 False。
                        left_bboxes_bool = tf.reduce_all(
                            bboxes_iou_pred >= 0, axis=-1)

                        # 6.1.1 left_bboxes_pred 形状为(left_bboxes_quantity, 6)。
                        left_bboxes_pred = positives_one_pred[left_bboxes_bool]

                        # 6.1.2 求出剩余的 bboxes 数量，
                        # left_bboxes_quantity 是一个标量型张量。
                        left_bboxes_quantity = left_bboxes_pred.shape[0]

                        # 在 TF 2.4 版本时，如果没有剩余的 bboxes，会返回
                        # left_bboxes_quantity 一个 None，所以需要进行下面的转换。
                        # 而在 TF 2.9 时，似乎 left_bboxes_quantity 会直接等于 0。
                        if left_bboxes_quantity is None:
                            left_bboxes_quantity = 0

                        # 6.1.3 满足下面 2 个条件时，把没有命中标签的正样本 bboxes 也
                        # 记录下来。
                        if tf.math.logical_and(
                                (left_bboxes_quantity > 0),
                                (new_bboxes_quantity <
                                 Constants.BBOXES_PER_IMAGE.value)):

                            # left_bboxes_class_pred 是剩余 bboxes 的类别
                            # 预测结果，形状为 (left_bboxes_quantity,)。
                            left_bboxes_class_pred = left_bboxes_pred[:, 1]

                            left_bboxes_categories = tf.round(
                                left_bboxes_class_pred)
                            left_bboxes_categories_error = tf.abs(
                                left_bboxes_class_pred - left_bboxes_categories)
                            # 计算置信度 left_bboxes_confidence_pred 时，必须以
                            # 0.5 为阈值， 其形状为 (left_bboxes_quantity,)。
                            left_bboxes_confidence_pred = (
                                0.5 - left_bboxes_categories_error) / 0.5

                            # scenario_d_bboxes 是一个标量型张量。
                            scenario_d_bboxes = (new_bboxes_quantity +
                                                 left_bboxes_quantity)

                            # 6.3 如果 scenario_d_bboxes > BBOXES_PER_IMAGE，需
                            # 要对剩余的 bboxes，按类别置信度进行排序。
                            if (scenario_d_bboxes >
                                    Constants.BBOXES_PER_IMAGE.value):
                                # 6.3.1 left_bboxes_sorted_confidence 形状为
                                # (left_bboxes_quantity,)。
                                left_bboxes_sorted_confidence = tf.sort(
                                    left_bboxes_confidence_pred,
                                    direction='DESCENDING')

                                # 6.3.2 vacant_seats 是一个整数，表示还有多少个空位，
                                # 可以用于填充剩余的 bboxes。
                                vacant_seats = (
                                        Constants.BBOXES_PER_IMAGE.value -
                                        new_bboxes_quantity)

                                # 6.3.3 left_bboxes_confidence_pred 形状为
                                # (vacant_seats,)。
                                left_bboxes_confidence_pred = (
                                    left_bboxes_sorted_confidence[
                                        : vacant_seats])

                            # left_bboxes_ious_pred 形状为 (vacant_seats,)，
                            # 或者是 (left_bboxes_quantity,)。
                            left_bboxes_ious_pred = tf.zeros_like(
                                left_bboxes_confidence_pred)

                            # 6.4 left_positive_bboxes_pred 形状为
                            # (vacant_seats, 2) 或 (left_bboxes_quantity, 2)。
                            left_positive_bboxes_pred = tf.stack(
                                values=[left_bboxes_confidence_pred,
                                        left_bboxes_ious_pred], axis=1)

                            # 6.5 记录剩余 bboxes 信息。append_left_bboxes
                            # 形状为 (BBOXES_PER_IMAGE +
                            # left_bboxes_quantity/vacant_seats, 2)。
                            append_left_bboxes = tf.concat(
                                values=[one_image_positive_bboxes,
                                        left_positive_bboxes_pred],
                                axis=0)

                            # 6.6 one_image_positive_bboxes，形状为
                            # (BBOXES_PER_IMAGE, 2)。
                            one_image_positive_bboxes = (
                                append_left_bboxes[
                                    -Constants.BBOXES_PER_IMAGE.value:])

                    # 更新最后一个状态量 latest_positive_bboxes。 形状为 (CLASSES,
                    # LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)。
                    self.latest_positive_bboxes[category, 1:].assign(
                        self.latest_positive_bboxes[category, :-1])

                    # latest_positive_bboxes 形状为 (Constants.CLASSES.value,
                    # LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)。
                    self.latest_positive_bboxes[category, 0].assign(
                        one_image_positive_bboxes)

    # noinspection PyMethodMayBeStatic
    def result(self):
        """对于当前所有已出现类别，使用状态值 state，计算 mean average precision。"""
        # 不能直接使用 tf.Variable 进行索引，需要将其转换为布尔张量。
        # showed_up_classes 形状为 (Constants.CLASSES.value,)。
        showed_up_classes_tensor = tf.convert_to_tensor(
            self.showed_up_classes, dtype=tf.bool)

        # average_precision_per_iou 形状为 (10,)。
        average_precision_per_iou = tf.zeros(shape=(10,))
        # 把 10 个不同 IoU 阈值情况下的 AP，放入张量 average_precision_per_iou
        # 中，然后再求均值。
        for iou_threshold in tf.linspace(0.5, 0.95, num=10):

            # average_precisions 形状为 (80,)，存放的是每一个类别的 AP。
            average_precisions = tf.zeros(
                shape=(Constants.CLASSES.value,))
            # 对所有出现过的类别，将其 AP 放入 average_precisions 中，然后再求均值。
            for category in range(Constants.CLASSES.value):

                # 只使用出现过的类别计算 AP。
                if self.showed_up_classes[category]:
                    # 1. 计算 recall_precisions。
                    recall_precisions = tf.ones(shape=(1,))
                    true_positives = tf.constant(0., shape=(1,))
                    false_positives = tf.constant(0., shape=(1,))

                    # 下面按照类别置信度从大到小的顺序，对 bboxes 进行排序。
                    # positive_bboxes_category 形状为
                    # (LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)
                    positive_bboxes_category = self.latest_positive_bboxes[
                        category]

                    # positive_bboxes_category 形状为
                    # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE, 2)
                    positive_bboxes_category = tf.reshape(
                        positive_bboxes_category, shape=(-1, 2))

                    # confidence_category 形状为
                    # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE,)。
                    confidence_category = positive_bboxes_category[:, 0]

                    # sorted_classification_confidence 形状为
                    # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE,)。
                    sorted_classification_confidence = tf.argsort(
                        values=confidence_category,
                        axis=0, direction='DESCENDING')

                    # sorted_bboxes_category 形状为
                    # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE, 2)。
                    sorted_bboxes_category = tf.gather(
                        params=positive_bboxes_category,
                        indices=sorted_classification_confidence, axis=0)

                    # 一个奇怪的事情是，使用 for bbox in sorted_bboxes_category，
                    # 它将不允许对 recall_precisions 使用 tf.concat。
                    # 下面更新 recall_precisions。
                    for i in range(len(sorted_bboxes_category)):
                        bbox = sorted_bboxes_category[i]
                        # sorted_bboxes_category 中，有一些是空的 bboxes，是既
                        # 没有标签，也没有预测结果。当遇到这些 bboxes 时，说明已经遍
                        # 历完预测结果，此时应跳出循环。空的 bboxes 类别置信度为 0.
                        bbox_classification_confidence = bbox[0]

                        if bbox_classification_confidence > 0:
                            bbox_iou = bbox[1]
                            # 根据当前的 iou_threshold，判断该 bbox 是否命中标签。
                            if bbox_iou > iou_threshold:
                                true_positives += 1
                                # 如果增加了一个 recall ，则记录下来。
                                recall_increased = True
                            else:
                                false_positives += 1
                                recall_increased = False

                            # 计算精度 precision。
                            precision = true_positives / (true_positives +
                                                          false_positives)

                            # recall_precisions 形状为 (x,)。如果有新增加了一个
                            # recall，则增加一个新的精度值。反之如果 recall 没有
                            # 增加，则把当前的精度值更新即可。
                            recall_precisions = tf.cond(
                                pred=recall_increased,
                                true_fn=lambda: tf.concat(
                                    values=[recall_precisions, precision],
                                    axis=0),
                                false_fn=lambda: tf.concat(
                                    values=[recall_precisions[:-1],
                                            precision], axis=0))

                    # 2. 计算当前类别的 AP。使用累加多个小梯形面积的方式来计算 AP。

                    # labels_quantity 是当前类别中，所有标签的总数。
                    labels_quantity = tf.math.reduce_sum(
                        self.labels_quantity_per_image[category])

                    # update_state 方法中区分了 a,b,c,d 共 4 种情况，scenario_d
                    # 属于下面这种，即有预测结果和标签，需要计算 AP 的情况。
                    # 如果有标签，即 labels_quantity > 0，要计算 AP。
                    if labels_quantity > 0:

                        # trapezoid_height 是每一个小梯形的高度。
                        # 注意！！！如果没有标签也计算小梯形高度，trapezoid_height
                        # 将会是 inf，并最终导致 NaN。所以要设置
                        # labels_quantity > 0.
                        trapezoid_height = 1 / labels_quantity

                        # accumulated_edge_length 是每一个小梯形的上下边长总和。
                        # accumulated_edge_length = 0.
                        accumulated_edge_length = tf.constant(
                            0., dtype=tf.float32)

                        # recalls 是总的 recall 数量。因为第 0 位并不是真正的
                        # recall，所以要减去 1.
                        recalls = len(recall_precisions) - 1

                        if recalls == 0:
                            # scenario_b 是有标签但是没有预测结果，包括在这种情况
                            # recalls==0，累计的梯形面积应该等于 0，AP 也将等于0。
                            accumulated_area_trapezoid = tf.constant(
                                0, dtype=tf.float32)

                        else:
                            for i in range(recalls):
                                top_edge_length = recall_precisions[i]
                                bottom_edge_length = recall_precisions[
                                    i + 1]

                                accumulated_edge_length += (
                                        top_edge_length +
                                        bottom_edge_length)

                            # 计算梯形面积：(上边长 + 下边长) * 梯形高度 / 2 。
                            accumulated_area_trapezoid = (
                                accumulated_edge_length *
                                trapezoid_height) / 2

                    # 而如果没有标签，则 average_precision=0。
                    # accumulated_area_trapezoid 就是当前类别的
                    # average_precision。scenario_c 属于这种情况。
                    else:
                        accumulated_area_trapezoid = tf.constant(
                            0, dtype=tf.float32)

                    # 构造索引 category_index，使它指向当前类别。
                    category_index = np.zeros(
                        shape=(Constants.CLASSES.value,))
                    category_index[category] = 1
                    # category_index 形状为 (Constants.CLASSES.value,)。
                    category_index = tf.convert_to_tensor(category_index,
                                                          dtype=tf.bool)

                    # average_precisions 形状为 (Constants.CLASSES.value,)。
                    average_precisions = tf.where(
                        condition=category_index,
                        x=accumulated_area_trapezoid[tf.newaxis],
                        y=average_precisions)

            # 把出现过的类别过滤出来，用于计算 average_precision。
            # average_precision_showed_up_categories 形状为 (x,)，即一共有 x
            # 个类别出现过，需要参与计算 AP。
            average_precision_showed_up_categories = average_precisions[
                showed_up_classes_tensor]

            # 一种特殊情况是，没有标签，预测结果也没有任何物体，此时不可以计算均值，否
            # 则会得出 NaN。这时直接把均值设置为 0 即可。
            if len(average_precision_showed_up_categories) != 0:
                # average_precision_over_categories 形状为 (1,)。
                average_precision_over_categories = tf.math.reduce_mean(
                    average_precision_showed_up_categories, keepdims=True)

            else:
                # 因为下面要进行拼接操作 concat，所以应该设置形状 (1,)，否则会出错。
                average_precision_over_categories = tf.constant(0.,
                                                                shape=(1,))

            # average_precision_per_iou 形状始终保持为 (10,)。
            average_precision_per_iou = tf.concat(
                values=[average_precision_per_iou[1:],
                        average_precision_over_categories], axis=0)

        mean_average_precision = tf.math.reduce_mean(
            average_precision_per_iou)

        return mean_average_precision

    # noinspection PyMethodMayBeStatic
    def reset_state(self):
        """每个 epoch 开始时，需要重新把状态初始化。"""
        self.latest_positive_bboxes.assign(
            tf.zeros_like(self.latest_positive_bboxes))

        self.labels_quantity_per_image.assign(
            tf.zeros_like(self.labels_quantity_per_image))

        self.showed_up_classes.assign(tf.zeros_like(self.showed_up_classes))


class SaveModelHighestAP(keras.callbacks.Callback):
    """指标 AP 达到最高时，保存模型。

    因为计算 AP 指标花的时间很长，并且 AP 指标的计算图太大，个人 PC 上无法构建计算图，所以单
    独创建一个评价模型，在运行一定 epochs 次数之后，才计算一次 AP 指标，并且是在 eager 模式
    下计算。而训练模型本身则始终在图模式下进行，这样既能保证模型运行速度，又能实现 AP 指标
    的计算。具体 5 个操作如下：
    1. 先创建一个专用的模型 evaluation_ap，用于计算指标 AP。
    2. 当训练次数 epochs 满足 2 个条件时，把当前训练模型 self.model 的权重提取出来，可以
        用 get_weights。
        epochs 需要满足的 2 个条件是：
        a. epochs ≥ epochs_warm_up，epochs_warm_up 是一个整数，表示经过一定数量的训
        练之后才保存权重。
        b. epochs % skip_epochs == 0，表示每经过 skip_epochs 个 epochs 之后，
        提取权重。如果设置 skip_epochs = 1，则表示每个 epoch 都会提取权重。
    3. 把提取到的权重，用 set_weights 加载到指标测试模型 evaluation_ap 上，然后用模型
        evaluation_ap 来测试指标。
    4. 如果指标 AP 大于最好的 AP 记录，且提供了保存模型的路径，则把该模型保存为
        highest_ap_model。
    5. 如果提供了路径 ongoing_training_model_path，还可以保存当前正在训练的模型。

    Attributes:
        evaluation_data: 用来计算 AP 的输入数据，一般应该使用验证集数据。
        highest_ap_model_path: 一个字符串，是保存最高 AP 指标模型的路径。如果为 None，
            则不保存 AP 指标最高的模型。
        evaluation_model: 一个 Keras 模型，专门用于计算 AP 指标。
        coco_average_precision: 是自定义类 MeanAveragePrecision 的个体 instance，
            用于计算 AP 指标。
        epochs_warm_up: 一个整数，表示从多少个 epochs 训练之后，开始计算 AP 指标。
        skip_epochs: 一个整数，表示每经过多少个 epochs，计算一次 AP 指标。
        ap_record: 一个列表，用于记录所有的 AP 指标。
        ap_threashold: 一个浮点数阈值。只有当 AP 大于该阈值时，才保存模型。
        ongoing_training_model_path: 一个字符串，是每个 epoch 之后保存当前模型的路径。
            如果为 None，则不保存当前模型。
    """

    def __init__(self, evaluation_data, highest_ap_model_path=None,
                 epochs_warm_up=100, skip_epochs=50, ap_threashold=None,
                 ongoing_training_model_path=None):
        super().__init__()
        self.evaluation_data = evaluation_data
        self.highest_ap_model_path = highest_ap_model_path

        self.coco_average_precision = MeanAveragePrecision()

        self.epochs_warm_up = epochs_warm_up
        self.skip_epochs = skip_epochs
        self.ap_threashold = ap_threashold
        self.ongoing_training_model_path = ongoing_training_model_path

    # noinspection PyUnusedLocal, PyAttributeOutsideInit
    def on_train_begin(self, logs=None):
        """因为每一次新的训练开始时，可能使用了不同的损失函数和超参，所以重新创建评价模型，
        并对其进行编译。还要把 ap_record 清零，从头开始一次新的记录。
        """
        custom_objects = {'MishActivation': MishActivation,
                          'ExtractImagePatches': ExtractImagePatches,
                          'ClipWeight': ClipWeight,
                          'PositionEncoding': PositionEncoding}
        with keras.utils.custom_object_scope(custom_objects):
            config = self.model.get_config()
            # 创建模型时，直接使用当前模型的 config，这样可以避免重复调用创建模型的各种参数。
            self.evaluation_model = keras.Model.from_config(config)

        # 虽然损失函数是自定义的，但是它不是类 class，所以下面可以直接用 self.model.loss
        # 来调用这个损失函数，并且会把它的参数值自动带过来。而其它自定义的类，则必须放到
        # custom_objects 中。
        self.evaluation_model.compile(
            run_eagerly=True,  # 1.6 在 eager 模式下计算 AP 指标。
            metrics=[self.coco_average_precision],
            loss=self.model.loss,
            optimizer=self.model.optimizer)  # self.model.optimizer 是当前训练模型的优化器。

        if self.ap_threashold is not None:
            self.ap_record = [self.ap_threashold]  # 设置 AP 阈值。
        else:
            self.ap_record = []  # 每次新的训练开始时，都要进行一次清零。

    # noinspection PyUnusedLocal
    def on_epoch_end(self, epoch, logs=None):
        """在一个 epoch 结束之后，计算 AP，并保存最高 AP 对应的模型。"""
        # 先保存一个持续训练的模型。
        if self.ongoing_training_model_path is not None:
            self.model.save(self.ongoing_training_model_path)

        # 使用 (epoch + 1) 是因为 epoch 是从 0 开始计算的，但实际上 epoch = 0 时，
        # 就已经是第一次迭代了。
        if tf.logical_and(
            ((epoch + 1) >= self.epochs_warm_up),
                ((epoch + 1 - self.epochs_warm_up) % self.skip_epochs == 0)):

            # 先获取当前模型的权重。
            current_weights = self.model.get_weights()
            # 把当前模型的权重，应用到模型 evaluation_ap 上。
            self.evaluation_model.set_weights(current_weights)

            # 因为没有对 evaluation_model 进行训练，所以 AP 指标不会在每个 epoch 之后
            # 自动复位，指标的状态量还存储着上一个 epoch 的状态，必须手动用
            #  reset_state() 对指标进行复位，否则会发生计算错误。
            self.coco_average_precision.reset_state()

            if not self.ap_record:
                max_ap_record = 0  # 列表为空时，无法使用 amax 函数，所以直接赋值为 0.
            else:
                max_ap_record = np.amax(self.ap_record)

            print(f'\nChecking the AP after epoch {epoch + 1}. The highest '
                  f'AP is: {max_ap_record:.2%}')

            evaluation = self.evaluation_model.evaluate(self.evaluation_data)
            # evaluation 是一个列表，包括 1 个损失值和 1 个 AP。
            current_ap = evaluation[-1]

            if current_ap > max_ap_record:
                print(f'The highest AP changed to: {current_ap:.2%}')
                if self.highest_ap_model_path is not None:
                    self.model.save(self.highest_ap_model_path)
                    print(f'The highest AP model is saved.')

            # 在上面取得 max_ap_record 之后，才把指标加到列表 ap_record 中去，
            # 否则程序逻辑会出错。
            self.ap_record.append(current_ap)


def _visualize_one_batch_prediction(
        images_batch, bboxes_batch,
        objectness_threshold,
        classification_threshold,
        show_classification_confidence=True,
        categories_to_detect=None, enlarged_image_scale=1,
        is_image=True, is_video=False):
    """将 Vision Transformer Detector 一个批次的预测结果显示为图片。

    Arguments:
        images_batch：一个图片张量，形状为 (batch_size, *MODEL_IMAGE_SIZE, 3)。
        bboxes_batch：一个 float32 型数组，代表所有保留下来的 bboxes，形状为
            (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。其中每个长度为 6 的向量，第 0 为类别
            的 id，第 1 位是类别置信度，后四位是 bbox 的参数，分别是
            (center_x, center_y, height_bbox, width_bbox)。
        objectness_threshold: 一个 [0, 1] 之间的浮点数，是物体框内是否有物体的置信度
            阈值。如果预测结果的置信度小于该值，则对应的探测框不显示。
        classification_threshold: 一个 [0, 1] 之间的浮点数，是对于物体框内的物体，属于
            某个类别的置信度。如果预测结果的置信度小于该值，则对应的探测框不显示。
        show_classification_confidence：一个布尔值。如果该值为 False，此时在输出的图
            片中，只显示类别的名字，不显示类别置信度。
        categories_to_detect：一个 Pandas 的 DataFrame 对象，包括了所有需要探测的类别。
        enlarged_image_scale：一个浮点数，表示将图片放大显示的倍数。
        is_image：一个布尔值。如果用户输入图片，则该值为真，否则为假。
        is_video：一个布尔值。如果用户输入视频，则该值为真，否则为假。

    Returns:
        image_bgr: 对输入的每一个图片，都显示 3 个探测结果图片。3 个结果图片对应 p5, p4
            和 p3 共 3 个不同的特征层，结果图片内显示探测到的物体。
            如果在某个特征层没有探测到物体，则不显示该特征层的图片。
    """

    # 留给显示信息的高度和宽度。
    text_height = 20
    text_width = 60

    image_bgr = None
    for image_tensor, bboxes in zip(images_batch, bboxes_batch):

        # 此时 image_tensor 的数值范围是 [-1, 1]，需要转换为 [0, 255]
        image_tensor += 1
        image_tensor *= 127.5
        image_pillow = keras.preprocessing.image.array_to_img(
            image_tensor)
        # 将图片进行放大，输入给 ImageOps.pad 的图片大小必须是整数。
        enlarged_height = round(image_tensor.shape[0] * enlarged_image_scale, 0)
        enlarged_width = round(image_tensor.shape[1] * enlarged_image_scale, 0)
        # noinspection PyUnresolvedReferences
        image_pillow = PIL.ImageOps.pad(
            image_pillow, (int(enlarged_width), int(enlarged_height)),
            centering=(0.5, 0.5))

        # 因为 OpenCV 要求图片数组的数据类型为 uint8 。需要对数据类型进行转换，
        # np.asarray 会将图片对象默认生成 uint8 类型的数据。
        array_rgb = np.asarray(image_pillow, dtype=np.uint8)

        # 注意这里要用 numpy 数组的深度拷贝。因为后续的画框和写字操作，会对数组进行修改，
        # 所以无法直接在原始数组上进行，需要用 copy 新建一个图片数组。
        # 同时将 rgb 通道调整为 OpenCV 的 bgr 通道
        image_bgr = array_rgb[..., [2, 1, 0]].copy()

        if image_bgr is None:
            sys.exit(f"Could not read the image_bgr in detection_results.")
        # 只需要 image_bgr 是3D张量即可。PIL，Keras 和 OpenCV 的图片数组，都是
        #  height 在前，width 在后。
        image_height, image_width = image_bgr.shape[: 2]

        image_bgr = array_rgb[..., [2, 1, 0]].copy()

        # 如果该图片有物体框，则显示这些物体框。如果没有物体框，则只显示图片。
        if bboxes is not None:
            # 遍历每一个物体框。each_bbox 是一个长度为 6 的向量，其中第 0 位是置信度，
            # 第 1 位是类别，后面 4 位是物体框信息。
            for each_bbox in bboxes:
                objectness_confidence = each_bbox[0]

                # 如果物体框内没有物体，应该停止检查当前物体框的信息，直接检查下一个物体框。
                if objectness_confidence < objectness_threshold:
                    continue

                # id_in_model_float 是一个浮点数，需要将它转换为整数。
                id_in_model_float = each_bbox[1]
                # id_in_model 是一个相当于整数的浮点数（它以 .0 结尾）。
                # 注意它是以 0.5 为分界线进行取整，而不是四舍五入的方式。
                id_in_model = np.round(id_in_model_float)

                category_name = categories_to_detect.at[id_in_model, 'name']

                # 注意 class_confidence 的计算方式：因为两个类别之间的差距为 1，所以当
                # 预测值和类别 id 的差距为 0.5 时，认为此时类别置信度为 0。并且预测值和
                # 类别 id 的差距越小，class_confidence越大。
                classification_error = np.abs(id_in_model - id_in_model_float)
                class_confidence = (0.5 - classification_error) / 0.5

                # 如果物体框内没有物体，应该停止检查当前物体框的信息，直接检查下一个物体框。
                if class_confidence < classification_threshold:
                    continue

                class_confidence = f'{class_confidence:.0%}'

                if show_classification_confidence:
                    show_text = category_name + ' ' + class_confidence

                # 如果是标签，则不需要显示置信度，因为此时置信度完全为 100%。
                else:
                    show_text = category_name

                bbox_center_x = each_bbox[-4] * enlarged_image_scale
                bbox_center_y = each_bbox[-3] * enlarged_image_scale
                bbox_height = each_bbox[-2] * enlarged_image_scale
                bbox_width = each_bbox[-1] * enlarged_image_scale

                # 此处获得预测框和真实的坐标。
                top_left_point_x = bbox_center_x - bbox_width / 2
                top_left_point_y = bbox_center_y - bbox_height / 2
                top_left_point_x = int(top_left_point_x)
                top_left_point_y = int(top_left_point_y)

                # 需要把物体框坐标限制在图片的范围之内，避免出现负数坐标。
                top_left_point_x = np.clip(top_left_point_x,
                                           a_min=0, a_max=image_width)
                top_left_point_y = np.clip(top_left_point_y,
                                           a_min=0, a_max=image_height)

                top_left_point = top_left_point_x, top_left_point_y

                bottom_right_point_x = bbox_center_x + bbox_width / 2
                bottom_right_point_y = bbox_center_y + bbox_height / 2
                bottom_right_point_x = int(bottom_right_point_x)
                bottom_right_point_y = int(bottom_right_point_y)

                # 需要把物体框坐标限制在图片的范围之内，避免出现负数坐标。
                bottom_right_point_x = np.clip(bottom_right_point_x,
                                               a_min=0, a_max=image_width)
                bottom_right_point_y = np.clip(bottom_right_point_y,
                                               a_min=0, a_max=image_height)

                bottom_right_point = (bottom_right_point_x,
                                      bottom_right_point_y)

                cv.rectangle(img=image_bgr, pt1=top_left_point,
                             pt2=bottom_right_point,
                             color=(0, 255, 0), thickness=2)

                text_point = list(top_left_point)
                text_point[1] -= 6
                if top_left_point[1] < text_height:
                    text_point[1] = text_height
                if image_width - top_left_point[0] < text_width:
                    text_point[0] = image_width - text_width

                cv.putText(img=image_bgr, text=show_text, org=text_point,
                           fontFace=cv.FONT_HERSHEY_TRIPLEX,
                           fontScale=0.5, color=(0, 255, 0))

        if is_image:
            print('\nPress key "q" to close all image windows.\n'
                  'Press key "s" to save the detected image.')
            cv.imshow(f'detection result', image_bgr)
            keyboard = cv.waitKey(0)
            if keyboard == ord("s"):
                cv.imwrite(r'tests\image_test.png', image_bgr)
            if keyboard == ord('q') or keyboard == 27:  # 27 为 esc 键
                cv.destroyAllWindows()

            print('\nPress any key to close all image windows.')
            cv.waitKey(0)
            cv.destroyAllWindows()
        elif is_video:
            # 可能需要在上一级的程序里显示图片。该部分内容待添加。
            # cv.imshow(f'{j}', image_bgr)
            pass

    return image_bgr


def visualize_predictions(
        image_input, predictions=None,
        objectness_threshold=None,
        classification_threshold=None,
        show_classification_confidence=True,
        categories_to_detect=None, enlarged_image_scale=1,
        is_image=True, is_video=False):
    """将 Vision Transformer 的预测结果显示为图片。

    Arguments:
        image_input: 是一个图片张量，形状为 (batch_size, *MODEL_IMAGE_SIZE, 3)。
            或者是一个 tf.data.dataset，包含如下 2 个元素，image_tensors 和 labels：
            image_tensors: 一个图片的 float32 型张量，形状为
                (batch_size, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3)。
            labels: 是一个 float32 型 3D 张量，形状为
                (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。
                最后 1 个维度大小为 6。这 6 个数值各自代表的含义是：
                第 0 位：是 0 或者 1，代表类别置信度。0 表示物体框内没有物体，1 表示物
                体框内有物体。
                第 1 位：是 [0, 79] 之间的整数，代表 80 个类别。如果物体框内没有物体，则
                该位数值等于 -8。
                最后 4 位：是物体框的位置和坐标，格式为 (x, y, height, width)，代表在
                图片中的实际大小，不是比例值。如果物体框内没有物体，则此 4 位数值等于 -8。
                其中 x 和 width 是 [0, MODEL_IMAGE_SIZE[1]] 之间的浮点数。
                其中 y 和 height 是 [0, MODEL_IMAGE_SIZE[0]] 之间的浮点数。
        predictions: 是图片对应的预测结果。如果 image_input 是 tf.data.Dataset，则
            predictions 可以缺省。
            如果 image_input 是一个图片张量，则必须有 predictions，并且 predictions
            应该是一个 tf.float32 类型的张量，张量形状为
            (batch_size, MAX_DETECT_OBJECTS_QUANTITY, 6)。
        objectness_threshold: 一个 [0, 1] 之间的浮点数，是物体框内是否有物体的置信度
            阈值。如果预测结果的置信度小于该值，则对应的探测框不显示。
        classification_threshold: 一个 [0, 1] 之间的浮点数，是对于物体框内的物体，属于
            某个类别的置信度。如果预测结果的置信度小于该值，则对应的探测框不显示。
        show_classification_confidence: 一个布尔值。如果该值为 False，此时在输出的图
            片中，只显示类别的名字，不显示类别置信度。
        categories_to_detect: 一个 Pandas 的 DataFrame 对象，包括了所有需要探测的类别。
        enlarged_image_scale: 一个浮点数，表示将图片放大显示的倍数。
        is_image: 一个布尔值。如果用户输入图片，则该值为真，否则为假。
        is_video: 一个布尔值。如果用户输入视频，则该值为真，否则为假。

    Returns:
        image_bgr: 对输入的每一个图片，都显示 3 个探测结果图片。3 个结果图片对应 p5, p4
            和 p3 共 3 个不同的特征层，结果图片内显示探测到的物体。
            如果在某个特征层没有探测到物体，则不显示该特征层的图片。
    """

    if objectness_threshold is None:
        objectness_threshold = Constants.OBJECTNESS_THRESHOLD.value
    if classification_threshold is None:
        classification_threshold = (
            Constants.CLASSIFICATION_CONFIDENCE_THRESHOLD.value)
    # ignore_types 是为了避免使用一些错误的类型，包括字符串或字节类型的数据。
    ignore_types = str, bytes

    # 如果没有 predictions，说明是标签数据。
    if predictions is None:
        # 如果是 tf.data.Dataset 的数据，则需要进行逐个取出。Dataset 属于 Iterable 。
        if isinstance(image_input, Iterable) and (
                not isinstance(image_input, ignore_types)):
            # 对于 Dataset 中的每一个 batch，对进行显示.
            for element in image_input:
                # 取出图片张量，形状为 (batch_size, *MODEL_IMAGE_SIZE, 3)
                images_batch = element[0]
                # 取出标签。
                labels_bboxes_batch = element[1]

                # 对于 Vision Transformer 模型，不需要进行 DIOU NMS 操作，直接显示。
                _visualize_one_batch_prediction(
                    images_batch=images_batch,
                    bboxes_batch=labels_bboxes_batch,
                    objectness_threshold=objectness_threshold,
                    classification_threshold=classification_threshold,
                    show_classification_confidence=False,
                    categories_to_detect=categories_to_detect,
                    enlarged_image_scale=enlarged_image_scale,
                    is_image=is_image, is_video=is_video)

    # 如果提供了 predictions，则默认为输入的是一个图片张量，以及一个预测结果元祖，其中包含
    #  3 个预测结果张量。
    else:
        # 需要对模型的输出结果进行解码，将其预测值转换为概率，并把 bbox 转换到 608x608
        # 图片中的等效大小。需要把预测结果使用 transform_predictions 进行转换。

        labels_bboxes_batch = transform_predictions(predictions)
        _visualize_one_batch_prediction(
            images_batch=image_input,
            bboxes_batch=labels_bboxes_batch,
            objectness_threshold=objectness_threshold,
            classification_threshold=classification_threshold,
            show_classification_confidence=show_classification_confidence,
            categories_to_detect=categories_to_detect,
            enlarged_image_scale=enlarged_image_scale,
            is_image=is_image, is_video=is_video)
