"""创建 tf.data.Dataset，专门用于物体探测的 Vision Transformer 模型。

Dataset 包含 2 个元素，分别为训练样本和标签。每一个标签中最多有 10 个物体信息，并且按每个
物体的 bbox 面积从大到小进行排列。
每一个物体的信息，由 6 位数字组成。6 位数字分别代表：
第 0 位，表示物体的类别 classification 概率。且概率值等于 1。
第 1 位，是一个整数，大小在 [0, 79] 区间，表示物体的类别。代表 COCO 数据集的 80 个类别。
第 2 位和第 3 位，分别代表物体框的中心点坐标 x 和 y。
第 4 位和第 5 位，分别代表物体框的高度和宽度。
中心点坐标 x 和物体框的宽度，数值的大小在 [0, width_image] 区间。
中心点坐标 y 和物体框的高度，数值的大小在 [0, height_image] 区间。
"""
import copy
import json
import os
import random

import numpy as np
import plotly
import plotly.graph_objects as go
from plotly import offline
from plotly.subplots import make_subplots
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# 导入自定义的全局变量和函数
from vision_transformer_detector import Constants

MODEL_IMAGE_HEIGHT = Constants.MODEL_IMAGE_SIZE.value[0]
MODEL_IMAGE_WIDTH = Constants.MODEL_IMAGE_SIZE.value[1]

# 在使用 Vision Transformer 探测器之前，需要手动设置下面 6 个模块常量 module constants。
# 模块常量的名字用大写字母，可以直接用于各个函数中。
# 1. 模块常量 CATEGORY_NAMES_TO_DETECT 是所有需要探测的类别的名字。
CATEGORY_NAMES_TO_DETECT = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

# 2. 模块常量 BBOX_AREA_DESCENDING: 一个布尔值。默认为 True，把每个图片的物体框，按照
# 面积从大到小进行排序。如果为 False，则从小到大排序。
BBOX_AREA_DESCENDING = True

# 3. 模块常量 PATH_IMAGE_TRAIN 是训练集图片的存放路径。
PATH_IMAGE_TRAIN = (r'D:\deep_learning\computer_vision\COCO_datasets'
                    r'\COCO_2017\train2017')
# 4. 模块常量 PATH_IMAGE_VALIDATION 是验证集图片的存放路径。
PATH_IMAGE_VALIDATION = (r'D:\deep_learning\computer_vision\COCO_datasets'
                         r'\COCO_2017\val2017')

# 5. 模块常量 instances_train2017.json 的路径。
TRAIN_ANNOTATIONS = (
    r'D:\deep_learning\computer_vision\COCO_datasets\COCO_2017'
    r'\annotations_trainval2017\annotations\instances_train2017.json')
# 6. 模块常量 instances_val2017.json 的路径。
VALIDATION_ANNOTATIONS = (
    r'D:\deep_learning\computer_vision\COCO_datasets\COCO_2017'
    r'\annotations_trainval2017\annotations\instances_val2017.json')

# COCO2017 有 80 个类别，但是因为部分 id 空缺，所以最大 id 编号为 90.
# COCO2017 中，最多标注数量的一张图片，标注了 93 个物体。但是标注数量的中位数，是 4 个标注。

# 模块常量   TRAIN_ANNOTATIONS_DICT 为训练集的标注文件，需要用函数
#  _get_annotations_dict_coco 进行生成。
TRAIN_ANNOTATIONS_DICT = {}
# TRAIN_ANNOTATIONS_RAW_DICT： 一个字典，是由 COCO 的标注文件转化而来。包含的5个键 key
# 为：'info', 'licenses', 'images', 'annotations', 'categories'。
TRAIN_ANNOTATIONS_RAW_DICT = {}


def _get_annotations_dict_coco(dataset_type):
    """把标注文件的信息转换为一个字典，并将该字典返回。字典的各个 key，就是图片编号，而 key
    对应的值 value，则是一个列表，该列表包含了此图片的所有标注。标注文件为 COCO 格式。

    Arguments:
        dataset_type： 一个字符串，指定是训练集 'train' 或者验证集 'validation'。

    Returns:
        annotations_dict： 一个字典，包含了 COCO 标注文件中，用于探测任务的所有标注信息。
            该字典的形式为：{'139':annotations_list_139,
            '285':annotations_list_285, ...}
            对于每一个有标注的图片，都会在字典 annotations_dict 中生成一个键值对。键
            key 是图片的编号，如上面的 '139'，值 value 则是一个标注列表，包含了该图片
            所有的标注。如上面的 annotations_list_139。
            每一个标注列表中，可以有若干个标注，并且每个标注也是一个列表，它的形式是：
            annotations_list_139 = [[annotation_1], [annotation_2], ...] 。所以，
            如果在 annotations_list_139 中包含20个列表，也就意味着图片 '139' 有 20 个
            标注。
            对于每一个标注，它的形式是 annotation_1=[category_id, center_point_x,
                center_point_y, height, width, bbox_area]
            其中，bbox_area 是探测框的面积。所以每个标注共有 6 个元素，标注的中间 4 位是
            探测框的中心点坐标，以及高度宽度。

        annotations_raw_dict: 一个字典，是从 COCO 的标注文件直接转化而来的原始字典。包含
            5个键：'info', 'licenses', 'images', 'annotations', 'categories'。

    """

    if dataset_type == 'train':
        path_annotations = TRAIN_ANNOTATIONS
    elif dataset_type == 'validation':
        path_annotations = VALIDATION_ANNOTATIONS
    else:
        path_annotations = None

    try:
        with open(path_annotations) as f:
            annotations_raw_dict = json.load(f)
    except FileNotFoundError:
        print(f'File not found: {path_annotations}')

    annotations_dict = {}
    # =======================做一个进度条。============================
    progress_bar = keras.utils.Progbar(
        len(annotations_raw_dict['annotations']), width=60, verbose=1,
        interval=0.5, stateful_metrics=None, unit_name='step')
    print(f'Extracting the annotations for {dataset_type} dataset ...')
    # =======================做一个进度条。============================

    # annotations_raw_dict['annotations'] 是一个列表，存放了所有的标注。
    # 而每个标注本身，则是一个字典，包含的key为：(['segmentation', 'area', 'iscrowd',
    # 'image_id', 'bbox', 'category_id', 'id'])
    fixed_records = []
    for i, each_annotation in enumerate(annotations_raw_dict['annotations']):
        progress_bar.update(i)

        # image_id 本身是一个整数，需要把它转换为字符，才能作为字典的key使用。
        image_id = str(each_annotation['image_id'])
        category_id = each_annotation['category_id']
        # bbox 是一个列表。
        bbox = each_annotation['bbox']
        top_left_x = bbox[0]
        top_left_y = bbox[1]
        # 注意标注文件中，bbox[2] 和 bbox[3] 是 bbox 的宽度和高度。
        width = bbox[2]
        height = bbox[3]

        center_point_x = top_left_x + width / 2
        center_point_x = round(center_point_x, 3)
        center_point_y = top_left_y + height / 2
        center_point_y = round(center_point_y, 3)

        # 有几张图片的高度或宽度为 0 ，直接将其设置为 1，以免遗漏该目标。（例如图片编号为
        # 200365 的图片，有根香肠的高度被标为了 0，需要将其改为 1.）
        if np.isclose(width, 0):
            width = 1
            one_record = ['Width', i, image_id, category_id,
                          center_point_x, center_point_y]
            fixed_records.append(one_record)
        elif np.isclose(height, 0):
            height = 1
            one_record = ['Height', i, image_id, category_id,
                          center_point_x, center_point_y]
            fixed_records.append(one_record)

        bbox_area = round(width * height, 1)

        # 需要把标注信息和图片对应起来，放到字典 annotations_dict 中。
        if image_id not in annotations_dict:
            # 如果还未记录该图片的任何标注信息，则先要把该图片建立为一个key。
            annotations_dict[image_id] = []
            first_annotation = [category_id, center_point_x, center_point_y,
                                height, width, bbox_area]
            annotations_dict[image_id].append(first_annotation)
        else:
            later_annotation = [category_id, center_point_x, center_point_y,
                                height, width, bbox_area]
            annotations_dict[image_id].append(later_annotation)

        # 这一部分检查是否有坐标值为负数的情况。
        if (bbox[1] < 0) or (bbox[0] < 0):
            print(f'Bbox error! Annotation index: {i}, image_id: {image_id}, '
                  f'category_id: {category_id}.\nIn "annotations" section: '
                  f'bbox coordinates are smaller than 0.\n'
                  f'bbox[0]: {bbox[0]}, bbox[1]: {bbox[1]}\n')

    for key, annotations in annotations_dict.items():
        # 把各个物体框，按照面积由大到小的顺序进行排序。
        if BBOX_AREA_DESCENDING:
            annotations_dict[key] = sorted(
                annotations, key=lambda annotation: annotation[-1],
                reverse=True)  # reverse=True 是从大到小排序。

        # 把各个物体框，按照面积从小到大排序。
        else:
            annotations_dict[key] = sorted(
                annotations, key=lambda annotation: annotation[-1],
                reverse=False)

    # 如果有错误信息，则输出所有错误信息。
    if len(fixed_records) > 0:
        print(f'\nDone. Here are {len(fixed_records)} fixed records.')
        for one_record in fixed_records:
            print(f'{one_record[0]} was 0, set to 1. '
                  f'\tImage: {one_record[2]}\tcategory_id: {one_record[3]},'
                  f'\tannotation index: {one_record[1]},\t'
                  f'object center {one_record[4]:.1f}, {one_record[5]:.1f}, ')

    return annotations_dict, annotations_raw_dict


# noinspection PyRedeclaration
TRAIN_ANNOTATIONS_DICT, TRAIN_ANNOTATIONS_RAW_DICT = _get_annotations_dict_coco(
    dataset_type='train')

VALIDATION_ANNOTATIONS_DICT = None
# noinspection PyRedeclaration
VALIDATION_ANNOTATIONS_DICT, _ = _get_annotations_dict_coco(
    dataset_type='validation')


# 以下为 2 个模块常量 module constants. 用函数 _coco_categories_to_detect 进行生成。
CATEGORIES_TO_DETECT = 0
FULL_CATEGORIES = 0


def _coco_categories_to_detect():
    """根据所要探测的类别名字 CATEGORY_NAMES_TO_DETECT，将类别名字、类别 id，类别所属
    的大类这 3 者关联起来，存储到一个 Pandas 的 DataFrame 中，并返回该 DataFrame。

    Returns:
        categories_to_detect: 一个 Pandas 的 DataFrame，包含了所有要探测的类别。
            表格里包括了一一对应的 4 类信息：id_in_model，id_in_coco，类别名字，以及该
            类别所属的大类。id_in_coco 是 COCO 中的 id 编号，id_in_model 是转换到模型
            中的编号，虽然 COCO 中只有 80 个类别，但是最大的 id 编号为 90，即最大的
            id_in_coco 为 90，而转换到模型中之后，最大的编号依然为 80，即最大的
            id_in_model 为 80。
        full_categories： 一个 Pandas 的 DataFrame。包含了 COCO 标注文件里所有 80 个
            类别，并且 id，类别名字和该类别所属的大类一一对应。

    """

    full_categories = pd.DataFrame({})
    for i, each in enumerate(TRAIN_ANNOTATIONS_RAW_DICT['categories']):
        id_in_model = i
        full_categories.loc[id_in_model, 'id_in_model'] = id_in_model
        full_categories.loc[id_in_model, 'id_in_coco'] = each['id']
        full_categories.loc[id_in_model, 'name'] = each['name']
        full_categories.loc[id_in_model, 'supercategory'] = each[
            'supercategory']

    categories = full_categories[
        full_categories['name'].isin(CATEGORY_NAMES_TO_DETECT)]

    categories_to_detect = categories.set_index('id_in_model')

    return categories_to_detect, full_categories


# CATEGORIES_TO_DETECT 中记录了需要探测的类别。注意对任意一个类别，它在模型中的 id 编号
# 始终不变，不论是在探测 5 个类别还是探测 80 个类别的情况。
# 举例来说，对于类别 toothbrush，它在模型内的编号将始终为 79（它在 COCO 中的编号为 90）。
# noinspection PyRedeclaration
CATEGORIES_TO_DETECT, FULL_CATEGORIES = _coco_categories_to_detect()


def _get_object_boxes_coco(one_image_path, image_original_size,
                           annotations_dict=None):
    """检查输入图片的所有标注，如果标注的类别在 CATEGORIES_TO_DETECT 中，则提取该标注对应
    的物体框。最终返回该输入图片中，所有需要探测类别的标注。

    Arguments:
        one_image_path： 一个字符串，是一张图片的完整路径。
        image_original_size: 是一个元祖，由 2 个元素组成，表示图片的初始大小，形式为
            (height, width)。
        annotations_dict: 是一个字典，包含了图片的所有标注文件。如果是训练集图片的标注
            文件，则不需要输入，直接使用模块常量 TRAIN_ANNOTATIONS_DICT。

    Returns:
        object_boxes: 一个列表，包括了一个图片中所有需要探测的物体框。列表的每一个元素
            代表一个物体框。
            每一个物体框，都是一个长度为 6 的元祖。6 位数字分别代表：
            第 0 位，表示物体的类别 classification 概率。且概率值等于 1。
            第 1 位，是一个整数，大小在 [0, 79] 区间，表示物体的类别。代表 COCO 数据集
            的 80 个类别。
            第 2 位和第 3 位，分别代表物体框的中心点坐标 x 和 y。
            第 4 位和第 5 位，分别代表物体框的高度和宽度。
            中心点坐标 x 和物体框的宽度，数值的大小在 [0, width_image] 区间。
            中心点坐标 y 和物体框的高度，数值的大小在 [0, height_image] 区间。。

    """
    # 如果 annotations_dict 为 None，则使用提前生成的训练集 TRAIN_ANNOTATIONS_DICT。
    if annotations_dict is None:
        annotations_dict = TRAIN_ANNOTATIONS_DICT

    # 因为是在 tf.py_function 下，即 eager 模式，所以下面可以使用 numpy()
    # COCO 2017 最大的图片名字是 000000581929.jpg
    image_name = str(one_image_path.numpy())[-15: -5]

    # 还要转换为整数，去掉 image_name 前面的 0，然后再转换回字符串类型
    image_name = int(image_name)
    image_name = str(image_name)

    # 这一部分先计算图片缩放之后产生的黑边 blank_in_height和 blank_in_width，后续
    # 计算探测框的坐标时会用到。
    image_original_height, image_original_width = image_original_size

    width_scale = image_original_width / MODEL_IMAGE_WIDTH
    height_scale = image_original_height / MODEL_IMAGE_HEIGHT
    resize_scale = None
    blank_in_height = 0
    blank_in_width = 0
    if width_scale > height_scale:
        resize_scale = width_scale
        resized_height = image_original_height / resize_scale
        blank_in_height = (MODEL_IMAGE_HEIGHT - resized_height) / 2
    elif width_scale == height_scale:
        resize_scale = width_scale
    elif width_scale < height_scale:
        resize_scale = height_scale
        resized_width = image_original_width / resize_scale
        blank_in_width = (MODEL_IMAGE_WIDTH - resized_width) / 2

    # image_annotations 是一个列表，包含了该图片的所有标注。并且是一个标志。
    # 因为有些图片没有任何标注，所以要用get方法，此时标志 image_annotations 将为空。
    image_annotations = annotations_dict.get(image_name)

    # 注意，因为后续的代码会把 image_annotations 清空，所以此处应该使用拷贝，
    # 重新生成一个列表。而且因为列表属于mutable，还必须使用 deepcopy，否则原字典
    # annotations_dict 也会被清空。
    image_annotations = copy.deepcopy(image_annotations)

    # labels 用于存放一个图片的所有标注，其中每一个标注都是一个长度为 6 的元祖。
    object_boxes = []

    while image_annotations:
        # 每一个标注是一个列表，含有 6 个元素，分别代表 (category_id_in_coco,
        # center_point_x, center_point_x, bbox_height, bbox_width, bbox_area)
        one_annotation = image_annotations.pop(0)

        id_in_coco = one_annotation[0]
        # 以下 if 语句，用于判断 id_in_coco 是否在表格 CATEGORIES_TO_DETECT 的中，即
        # 是否需要探测该类别。
        if (CATEGORIES_TO_DETECT['id_in_coco'].isin([id_in_coco])).any():
            # 根据表格 CATEGORIES_TO_DETECT，把 id_in_coco 转换为 id_in_model。
            category = CATEGORIES_TO_DETECT[
                CATEGORIES_TO_DETECT['id_in_coco'] == id_in_coco]
            # id_in_model 是 0 到 79 的整数，代表 80 个类别。
            id_in_model = int(category.index[0])  # 从 np.float64 转换为 int

            center_point_x = one_annotation[1]
            center_point_y = one_annotation[2]
            bbox_height = one_annotation[3]
            bbox_width = one_annotation[4]

            # 以下将坐标点和高宽转换为 608x608 大小图片中的实际值。
            center_point_x = center_point_x / resize_scale
            center_point_y = center_point_y / resize_scale
            bbox_height = bbox_height / resize_scale
            bbox_width = bbox_width / resize_scale

            # 原图在缩放到 608x608 大小并且居中之后，物体框中心点会发生移动。
            if width_scale >= height_scale:
                center_point_y += blank_in_height

            elif width_scale < height_scale:
                center_point_x += blank_in_width

            # 把 bbox 的 4 个参数，从 tf 张量转换为数值。
            center_point_x = center_point_x.numpy()
            center_point_y = center_point_y.numpy()
            bbox_height = bbox_height.numpy()
            bbox_width = bbox_width.numpy()
            # one_object_box 是一个元祖，长度为 6。第 0 位设置为 1， 表示该物体
            # 框中有物体；第 1 位是类别编码。最后 4 位是物体框信息。
            one_object_box = (1, id_in_model, center_point_x, center_point_y,
                              bbox_height, bbox_width)

            object_boxes.append(one_object_box)

    return object_boxes


def _get_paths_image_coco(path_image, images_range=None, shuffle_images=False):
    """从文件夹中提取图片，返回图片的名字列表。

    Arguments:
        path_image： 一个字符串，是所有图片的存放路径。
        images_range: 一个元祖，是图片的索引范围，如(0, 5000)。
        shuffle_images： 一个布尔值，如果为 True，则把全部图片的顺序打乱。

    Returns:
        paths_image: 一个列表，该列表是所有被选中图片的完整路径，例如：
            ['D:\\COCO_datasets\\COCO_2017\\train2017\\000000000009.jpg', ...]。
            列表的大小，由 images_range 设定。如不设定 images_range，将默认使用
            文件夹内全部图片。
    """

    paths_image = []
    # os.walk 会进入到逐个子文件中，找出所有的文件。
    for path, dir_names, image_names in os.walk(path_image):
        for image_name in image_names:
            one_image_path = os.path.join(path, image_name)
            paths_image.append(one_image_path)

    if shuffle_images:
        random.shuffle(paths_image)

    # 如果指定了图片范围，则只使用一部分的图片。
    if images_range is not None:
        start_index, end_index = images_range
        paths_image = paths_image[start_index: end_index]

    return paths_image


def _get_image_tensor_coco(one_image_path):
    """根据输入的图片路径，转换为一个 3D TF 张量并返回。

    Arguments:
        one_image_path： 一个字符串，是一张图片的完整路径。
    
    Returns:
        image_tensors: 一个图片的 3D 张量，形状为 (height, width, 3)。
        image_original_size: 是一个元祖，由2个元素组成，表示图片的初始大小，形式为
            (height, width)。后续根据标注文件生成标签时要用到这个列表里的信息。
    """

    image_file = tf.io.read_file(one_image_path)
    original_image_tensor = tf.image.decode_image(image_file, channels=3)

    # image_original_size 必须分开作为两个数处理，不能直接用shape[: 2]，否则会得到
    # 一个 shape 张量，而不是一个元祖
    image_original_size = (original_image_tensor.shape[0],
                           original_image_tensor.shape[1])

    image_tensor = tf.image.resize_with_pad(
        original_image_tensor, target_height=MODEL_IMAGE_HEIGHT,
        target_width=MODEL_IMAGE_WIDTH)

    # 将图片数值限制在 [0, 255] 范围。遥感图像的像素有负数。
    image_tensor = tf.clip_by_value(
        image_tensor, clip_value_min=0, clip_value_max=255)

    image_tensor /= 127.5  # 将图片转换为 [0, 2] 范围
    image_tensor -= 1  # 把图片转换为 [-1, 1] 范围。

    return image_tensor, image_original_size


def _get_label_arrays(one_image_path, image_original_size, dataset_type):
    """根据输入的图片路径，创建其对应的标签张量 labels。

    Arguments:
        one_image_path: 一个字符串，图片存放路径。
        image_original_size: 是一个元祖，由2个元素组成，表示图片的初始大小，形式为
            (height, width)。
        dataset_type: 一个字符串，指定是训练集 'train' 或者验证集 'validation'。

    Returns:
        labels: 是一个 float32 型 2D 张量，形状为 (MAX_DETECT_OBJECTS_QUANTITY, 6)。
                最后 1 个维度大小为 6。这 6 个数值各自代表的含义是：
                第 0 位：是 0 或者 1，代表类别置信度。0 表示物体框内没有物体，1 表示物
                体框内有物体。
                第 1 位：是 [0, 79] 之间的整数，代表 80 个类别。如果物体框内没有物体，则
                该位数值等于 -8。
                最后 4 位：是物体框的位置和坐标，格式为 (x, y, height, width)，代表在
                图片中的实际大小，不是比例值。如果物体框内没有物体，则此 4 位数值等于 -8。
                其中 x 和 width 是 [0, MODEL_IMAGE_SIZE[1]] 之间的浮点数。
                其中 y 和 height 是 [0, MODEL_IMAGE_SIZE[0]] 之间的浮点数。
    """

    # 使用已经提前生成的 ANNOTATIONS_DICT。
    if dataset_type == 'train':
        annotations_dict = TRAIN_ANNOTATIONS_DICT
    else:
        annotations_dict = VALIDATION_ANNOTATIONS_DICT

    # 根据要探测的类别 CATEGORIES_TO_DETECT，取出该图片的所有物体框。
    # object_boxes 是一个列表，列表中的每一个元素都是一个长度为 6 的元祖，该元祖包含了
    # 一个物体框的信息。
    object_boxes = _get_object_boxes_coco(
        one_image_path=one_image_path, image_original_size=image_original_size,
        annotations_dict=annotations_dict)

    # 保留前 MAX_DETECT_OBJECTS_QUANTITY 个物体框，使得 len(object_boxes) ==
    # MAX_DETECT_OBJECTS_QUANTITY。
    # noinspection PyTypeChecker
    if len(object_boxes) >= Constants.MAX_DETECT_OBJECTS_QUANTITY.value:
        labels = object_boxes[:
                              Constants.MAX_DETECT_OBJECTS_QUANTITY.value]
        # 注意这里必须把元祖 labels 转化为张量，再将其输出，否则后续会出现形状不匹配的问题，
        # 即便后面在 _wrapper 中使用了 labels.set_shape 也依然会出错。
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    else:
        # 把 labels 初始化，第 0 位等于 0，其它位置等于 -8.
        labels = np.ones(
            shape=(Constants.MAX_DETECT_OBJECTS_QUANTITY.value, 6)) * -8
        labels[:, 0] = 0
        # 只有当 object_boxes 非空时，即图片中至少有 1 个标注时，才把标注信息放入到数组
        # labels 中，否则会报错。
        if object_boxes:
            labels[: len(object_boxes)] = object_boxes

    return labels


def _wrapper(one_image_path, dataset_type):
    """根据输入的图片路径，转换为 2 个 TF 张量返回。

    使用 wrapper 的目的，是在 dataset.map 和实际需要使用的函数之间做一个过渡，这样实际
    需要使用的函数内才能使用 eager 模式。

    使用方法：
    dataset = dataset.map(
        lambda one_image_path: wrapper(one_image_path, dataset_type),
        num_parallel_calls=tf.data.AUTOTUNE)

    Arguments:
        one_image_path： 一个字符串，图片存放路径。

    Returns:
        image_tensors: 一个图片的 3D 张量，形状为 (MODEL_IMAGE_HEIGHT, 
            MODEL_IMAGE_WIDTH, 3)。
        labels: 是一个 float32 型 2D 张量，形状为 (MAX_DETECT_OBJECTS_QUANTITY, 6)。
            最后 1 个维度大小为 6。这 6 个数值各自代表的含义是：
            第 0 位：是 0 或者 1，代表类别置信度。0 表示物体框内没有物体，1 表示物
            体框内有物体。
            第 1 位：是 [0, 79] 之间的整数，代表 80 个类别。如果物体框内没有物体，则
            该位数值等于 -8。
            最后 4 位：是物体框的位置和坐标，格式为 (x, y, height, width)，代表在
            图片中的实际大小，不是比例值。如果物体框内没有物体，则此 4 位数值等于 -8。
            其中 x 和 width 是 [0, MODEL_IMAGE_SIZE[1]] 之间的浮点数。
            其中 y 和 height 是 [0, MODEL_IMAGE_SIZE[0]] 之间的浮点数。
    """
 
    image_tensor, image_original_size = tf.py_function(
        func=_get_image_tensor_coco, inp=[one_image_path],
        Tout=(tf.float32, tf.float32))

    labels = tf.py_function(
        func=_get_label_arrays,
        inp=[one_image_path, image_original_size, dataset_type],
        Tout=tf.float32)

    # 必须用 set_shape， 才能让 MapDataset 输出的张量有明确的形状。
    image_tensor.set_shape(shape=(*Constants.MODEL_IMAGE_SIZE.value, 3))
    labels.set_shape(
        shape=[Constants.MAX_DETECT_OBJECTS_QUANTITY.value, 6])

    return image_tensor, labels


def coco_data_vision_transformer(
        dataset_type, images_range=(0, 1000),
        shuffle_images=False, batch_size=8):
    """使用 COCO 2017 创建 tf.data.Dataset，用于 vision transformer 探测器。

    使用方法：
    train_dataset = coco_data_yolov4_csp(
        dataset_type='validation', images_range=(0, None), batch_size=16)

    Arguments:
        dataset_type： 一个字符串，指定是训练集 'train' 或者验证集 'validation'。
        images_range： 一个元祖，分别表示开始的图片和最终的图片。例如 (10, 50) 表示从
            第 10 张图片开始，到第 49 张图片结束。None 或者 (0, None) 表示使用全部图片。
        shuffle_images： 一个布尔值，是否打乱图片顺序。
        batch_size： 一个整数，是批次量的大小。

    Returns:
        返回一个 tf.data.dataset，包含如下 2 个元素，image_tensors 和 labels：
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
    """

    paths_image = {'train': PATH_IMAGE_TRAIN,
                   'validation': PATH_IMAGE_VALIDATION}

    # 获取训练集图片或是验证集图片的路径。
    path = paths_image.get(dataset_type, 'The input is invalid.')
    
    # image_paths 是一个列表，该列表包含了指定文件夹内，所有图片的完整路径。
    image_paths = _get_paths_image_coco(
        path_image=path, images_range=images_range,
        shuffle_images=shuffle_images)

    # 根据列表，生成 dataset。
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    # wrapper 将返回一个图片张量，和 1 个标签张量 labels。

    dataset = dataset.map(
        lambda one_image_path: _wrapper(one_image_path, dataset_type),
        num_parallel_calls=tf.data.AUTOTUNE)

    # 前面 dataset 的各个方法，用于处理单个文件，下面用 batch 方法生成批量数据。
    # drop_remainder 为 True 表示如果最后一批的数量小于 BATCH_SIZE，则丢弃最后一批。
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    # 预先取出一定数量的 dataset，用 tf.data.AUTOTUNE 自行调节数量大小。
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def scatters_plotly(scatters_inputs, titles, file_name,
                    secondary_y=False, secondary_y_title=None):
    """用 plotly 画一个折线图，图中可以有多条折线。
    如果使用 2 个不同的 y 坐标轴，则只有第一条曲线是用第一个 y 坐标轴，剩余其它曲线，将全部
    使用第二个 y 坐标轴。

    Arguments:
        scatters_inputs: 是一个列表，形式是 [(x1, y1, trace_name1), (x2, y2,
            trace_name2), ...]，列表内的每个元祖代表一个折线图。
            x 是一个序列（比如列表等），其中为若干浮点数，代表输入的横坐标值。
            y 是一个序列（比如列表等），其中为若干浮点数，代表纵坐标值，并且数量必须和 x
            的数量相同。
            trace_name，是一个字符串，字符串代表一条折线的名字。
        titles: 一个元祖，包含 3 个字符串，第一个字符串是整个折线图的名字，第二个字符串
            是横坐标名字，第三个字符串是纵坐标名字。
        file_name: 一个字符串，是这个折线图文件的名字，格式是 'xxx.html' 。
        secondary_y: 一个布尔值，如果为 True，则使用 2 个不同的 y 坐标轴。并且此时
            scatters_inputs 应该只有 2 条折线图。
        secondary_y_title: 一个字符串，是第二个 y 坐标轴的名字。只有 secondary_y 为
            True 时才会起作用。
    Returns:
        返回一个 html 格式的折线图文件，自动保存在当前文件夹中。
    """

    if secondary_y:
        fig = make_subplots(
            specs=[[{'secondary_y': secondary_y}]])
    else:
        fig = go.Figure()

    for i, each_scatter in enumerate(scatters_inputs):

        x = each_scatter[0]
        y = each_scatter[1]
        trace_name = each_scatter[2]

        if i == 0:
            fig.add_trace(go.Scatter(x=x, y=y, name=trace_name,
                                     mode='lines+markers'))
        elif secondary_y:
            fig.add_trace(go.Scatter(
                x=x, y=y, name=trace_name,
                mode='lines+markers'), secondary_y=True)
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y, name=trace_name, mode='lines+markers'))

    title = titles[0]
    xaxis_title = titles[1]
    yaxis_title = titles[2]
    fig.update_layout(title=title,
                      xaxis_title=xaxis_title,
                      yaxis_title=f'<b>{yaxis_title}</b>')
    if secondary_y:
        fig.update_yaxes(title_text=f'<b>{secondary_y_title}</b>',
                         secondary_y=True)

    offline.plot(fig, filename=file_name)
