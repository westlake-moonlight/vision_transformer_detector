"""该文件对比单进程和多进程条件下，代码的运行速度。如果只是需要查看 COCO 图片的统计值，
只需要使用下面的 coco_statistics_multi_processing 函数即可。

注意在使用多进程时，每一个进程都会把当前 Python 文件里的导入命令，全局变量都执行一遍。有时
这会导致一个奇怪的现象：如果导入命令和全局变量的创建会需要很多时间，那么每个进程也会需要很
多时间，最终导致多线程显得很慢。

解决对策是：把不需要的导入命令和全局变量，放入到一个“准备函数”中（如下面的 preparation()
函数），只要子进程没有用到“准备函数”，子进程就不会被拖慢。
"""
import json
from concurrent import futures
import functools
from collections import Counter
from pathlib import Path
import time

import numpy as np
import pandas as pd
from IPython.display import display


# 使用该 Python 文件之前，需要先设置好下面 5 个常量。
# 这 2 个路径是 COCO 数据集图片的路径。
PATH_IMAGE_TRAIN = Path(r'D:\deep_learning\computer_vision\COCO_datasets'
                        r'\COCO_2017\train2017')
PATH_IMAGE_VALIDATION = Path(r'D:\deep_learning\computer_vision\COCO_datasets'
                             r'\COCO_2017\val2017')
# 下面 2 个 JSON 文件和 1 个 csv 文件是我自己创建的文件，对 COCO 数据集进行统计时要用到。
PATH_TRAIN_ANNOTATIONS_DICT = Path(
    r'D:\deep_learning\computer_vision\COCO_datasets\COCO_2017'
    r'\coco_statistics\train_annotations_dict.json')
PATH_VAL_ANNOTATIONS_DICT = Path(
    r'D:\deep_learning\computer_vision\COCO_datasets\COCO_2017'
    r'\coco_statistics\val_annotations_dict.json')
PATH_FULL_CATEGORIES = Path(
    r'D:\deep_learning\computer_vision\COCO_datasets\COCO_2017'
    r'\coco_statistics\full_categories.csv')


def preparation():
    """多进程并发时，该函数的准备工作只需要做一遍。然后把需要用到的变量传递给各个子进程。"""

    try:
        with open(PATH_TRAIN_ANNOTATIONS_DICT, 'r') as f:
            train_annotations_dict = json.load(f)
        with open(PATH_VAL_ANNOTATIONS_DICT, 'r') as f:
            validation_annotations_dict = json.load(f)

        full_categories = pd.read_csv(PATH_FULL_CATEGORIES)
    except Exception as exc:
        print(f'While reading the files, raised:\n{exc}')
        full_categories = None  # 此行主要是为了避免 Pycharm 出现 warning。
    else:
        print('\nReading of JSON and dataframe is done!')

    return train_annotations_dict, validation_annotations_dict, full_categories


def worker(group_start_image: int,
           group_images_quantity: int,
           annotations_dict: dict,
           image_path: Path,
           ):
    """该函数是多进程并发时，单个进程的任务。
    对输入的每一张图片，统计 3 项内容：
        1. 一张图片内，最大的标注数量。
        2. 出现次数最多的类别，及其出现的次数。
        3. 在每张图片中，标注数量最多的类别。

    Arguments:
        image_path: 一个字符串，是 'train' 或者 'validation' 两者之一，表示两种数据集。
        group_start_image: 一个整数，表示从第 start_image 张图片开始进行统计。
        group_images_quantity: 一个正整数，是需要统计的图片总数。
        annotations_dict: 一个正整数，是需要统计的图片总数。
    """

    # annotations_tally 记录了每张图片内的标注数量，以及图片名字。
    group_annotations_tally = []
    group_max_annotations_in_one_image = pd.DataFrame()
    group_counter_annotation = 0
    # group_showed_up_category_in_all_images 是在被统计的图片中，所有出现过的类别编号列表。
    group_showed_up_category_in_all_images = []

    for j, each in enumerate(image_path.iterdir()):
        # 从设定的第 start_image 张图片开始进行统计。
        if j >= group_start_image:
            image_name_int = int(each.stem)  # 去掉名字前面的多个数字 0。
            image_name = str(image_name_int)  # 需要将图片名转换为字符串。
            # annotations_in_one_image 是一个列表，代表图片里的所有标注内容。
            annotations_in_one_image = annotations_dict.get(image_name, [])
            # annotation_tally 是一个图片内的标注数量。
            annotation_tally = len(annotations_in_one_image)
            group_annotations_tally.append([annotation_tally, image_name_int])

            # 如果该图片内有标注，则统计其出现的类别，以及对应的数量。
            if annotations_in_one_image:
                # category_in_one_img 是一个图片内所有出现过的类别。
                category_in_one_img = []
                for each_annotation in annotations_in_one_image:
                    category_id_in_coco = each_annotation[0]
                    category_in_one_img.append(category_id_in_coco)

                # category_in_one_img_tally 是一个统计量，统计了所有出现的
                # 类别编号及其数量。
                category_in_one_img_tally = Counter(category_in_one_img)
                # max_annotations_each_image 是一张图片内，标注数量最高的类别编号
                # 及其数量。
                max_annotations_each_image = (
                    category_in_one_img_tally.most_common(1))

                id_in_coco = max_annotations_each_image[0][0]
                annotations_quantity = max_annotations_each_image[0][1]
                group_max_annotations_in_one_image.loc[
                    group_counter_annotation,
                    'category_id_in_coco'] = id_in_coco
                group_max_annotations_in_one_image.loc[
                    group_counter_annotation,
                    'annotations_quantity'] = annotations_quantity
                group_max_annotations_in_one_image.loc[
                    group_counter_annotation, 'image_name'] = image_name_int

                group_counter_annotation += 1

                # showed_up_category_in_one_img 是去掉了重复出现的类别。
                showed_up_category_in_one_img = set(category_in_one_img)
                group_showed_up_category_in_all_images += list(
                    showed_up_category_in_one_img)

            if j + 1 == group_start_image + group_images_quantity:
                break

    return (group_annotations_tally, group_max_annotations_in_one_image,
            group_showed_up_category_in_all_images)


# noinspection PyShadowingNames
def coco_statistics_multi_processing(
        datatype: str = 'train', start_image: int = 0,
        images_quantity: int = 80,
        multi_processing_threshold: int = 10_000) -> None:
    """使用多进程并发，查询训练集图片和验证集图片的统计数据。

    在被查询的图片中，统计 3 项内容：
    1. 一张图片内，最大的标注数量。
    2. 出现次数最多的类别，及其出现的次数。
    3. 在每张图片中，标注数量最多的类别。

    Arguments:
        datatype: 一个字符串，是 'train' 或者 'validation' 两者之一，表示两种数据集。
        start_image: 一个整数，表示从第 start_image 张图片开始进行统计。
        images_quantity: 一个正整数，是需要统计的图片总数。
        multi_processing_threshold: 一个正整数，每一组的图片数量大于此数时，使用
            多进程并发。

    """
    (TRAIN_ANNOTATIONS_DICT, VALIDATION_ANNOTATIONS_DICT,
     FULL_CATEGORIES) = preparation()

    if datatype == 'train':
        image_path = PATH_IMAGE_TRAIN
        annotations_dict = TRAIN_ANNOTATIONS_DICT
    elif datatype == 'validation':
        image_path = PATH_IMAGE_VALIDATION
        annotations_dict = VALIDATION_ANNOTATIONS_DICT
    else:
        print('Only "train" or "validation" is valid datatype.\n')
    print(f'从第 {start_image:,} 张图片开始，统计 {images_quantity:,} 张图片:\n')

    # annotations_tally 记录了每张图片内的标注数量，以及图片名字。
    annotations_tally = []
    max_annotations_in_one_image = pd.DataFrame({})
    # showed_up_category_in_all_images 是在被统计的图片中，所有出现过的类别编号列表。
    showed_up_category_in_all_images = []

    if images_quantity > multi_processing_threshold:
        # processes_quantity 是分组的数量。
        processes_quantity = int(
            np.ceil(images_quantity / multi_processing_threshold))
        # group_size 是每一组的图片数量。
        group_images_quantity = int(
            np.ceil(images_quantity / processes_quantity))
        print(f'Python processes: {processes_quantity:3}'
              f'\ngroup_images_quantity: {group_images_quantity:8_}')

        # noinspection PyUnboundLocalVariable
        sub_process_worker = functools.partial(
            worker,
            image_path=image_path,
            annotations_dict=annotations_dict,
        )
        # group_start_images 是一个列表，列表元素是每一组起始图片的名字。
        group_start_images = [start_image + i * group_images_quantity
                              for i in range(processes_quantity)]

        # 因为最后一组的图片数量，可能和其它组不同，所以要进行专门处理，并使用
        # group_images_quantity_list 进行记录。
        group_images_quantity_list = []
        left_images = images_quantity % group_images_quantity
        if left_images == 0:
            for j in range(processes_quantity):
                group_images_quantity_list.append(group_images_quantity)
        else:
            # 前面 processes_quantity - 1 组，每组的图片数量相同。
            for j in range(processes_quantity - 1):
                group_images_quantity_list.append(group_images_quantity)
            # 最后一组，只需要把剩余的图片加进去即可。
            group_images_quantity_list.append(left_images)

        print(f'group_start_images:\t{group_start_images}')
        print(f'group_images_quantity_list:\t{group_images_quantity_list}\n')

        with futures.ProcessPoolExecutor(
                max_workers=processes_quantity) as executor:

            # map 方法返回一个生成器。
            res = executor.map(sub_process_worker,
                               group_start_images, group_images_quantity_list)

        # 遍历结果生成器，把每一组的结果汇总起来。
        for (group_annotations_tally, group_max_annotations,
             group_showed_up_category) in res:

            annotations_tally += group_annotations_tally
            max_annotations_in_one_image = pd.concat(
                [max_annotations_in_one_image, group_max_annotations], axis=0)
            showed_up_category_in_all_images += group_showed_up_category

    else:
        # noinspection PyUnboundLocalVariable
        (group_annotations_tally, group_max_annotations,
         group_showed_up_category) = worker(
            group_start_image=start_image,
            group_images_quantity=images_quantity,
            image_path=image_path, annotations_dict=annotations_dict)

        annotations_tally += group_annotations_tally
        max_annotations_in_one_image = pd.concat(
            [max_annotations_in_one_image, group_max_annotations], axis=0)
        showed_up_category_in_all_images += group_showed_up_category

    # annotations_array 形状为 (x, 2)，x 为统计图片的数量。
    annotations_array = np.asarray(annotations_tally)
    to_sort_annotations = annotations_array[:, 0]
    sorted_indexes = np.argsort(to_sort_annotations)
    # sorted_indexes 是排列后的索引，按照标注数量，从大到小排列。
    sorted_indexes = np.flip(sorted_indexes)
    # 把 annotations_array 按照标注数量，从大到小排列。
    annotations_array = annotations_array[sorted_indexes]
    print(f'一张图片内的最大标注数量:  {annotations_array[0, 0]}，  '
          f'该图片的名字编号为： {str(annotations_array[0, 1])}')

    showed_up_category_in_all_images_tally = Counter(
        showed_up_category_in_all_images)
    # max_showed_up_category_in_all_images 是所有被统计图片中，出现次数最多的类别。
    max_showed_up_category_in_all_images = (
        showed_up_category_in_all_images_tally.most_common(1))
    # max_showed_up_id_in_coco 是在 COCO 数据集中的类别 ID。
    max_showed_up_id_in_coco = max_showed_up_category_in_all_images[0][0]
    max_showed_up_times = max_showed_up_category_in_all_images[0][1]
    max_showed_up_row = FULL_CATEGORIES[
        FULL_CATEGORIES['id_in_coco'].isin([max_showed_up_id_in_coco])]
    # max_showed_up_name 是在 COCO 数据集中的类别名称。
    max_showed_up_name = FULL_CATEGORIES.at[max_showed_up_row.index[0], 'name']
    print(f'出现次数最多的类别编号:  {max_showed_up_id_in_coco}，  '
          f'该类别的名称为:  {max_showed_up_name}，  '
          f'该类别共出现在 {max_showed_up_times} 张图片中。')

    display_dataframe = max_annotations_in_one_image.set_index('image_name')
    display_dataframe = display_dataframe.sort_values(
        by='annotations_quantity', ascending=False)
    print(f'\n在每一张图片中，标注数量最多的类别记录如下（按照标注数量从大到小排列）:\n')
    display(display_dataframe)


# noinspection PyShadowingNames
def coco_statistics(datatype: str = 'train',
                    start_image: int = 0, images_quantity: int = 80) -> None:
    """该函数用于查询训练集图片和验证集图片的统计数据。使用单个 Python 进程。

    在被查询的图片中，统计 3 项内容：
    1. 一张图片内，最大的标注数量。
    2. 出现次数最多的类别，及其出现的次数。
    3. 在每张图片中，标注数量最多的类别。

    Arguments:
        datatype: 一个字符串，是 'train' 或者 'validation' 两者之一，表示两种数据集。
        start_image: 一个整数，表示从第 start_image 张图片开始进行统计。
        images_quantity: 一个正整数，是需要统计的图片总数。

    """

    (TRAIN_ANNOTATIONS_DICT, VALIDATION_ANNOTATIONS_DICT,
     FULL_CATEGORIES) = preparation()

    if datatype == 'train':
        image_path = PATH_IMAGE_TRAIN
        annotations_dict = TRAIN_ANNOTATIONS_DICT
    else:
        image_path = PATH_IMAGE_VALIDATION
        annotations_dict = VALIDATION_ANNOTATIONS_DICT
    print(f'从第 {start_image:,} 张图片开始，统计 {images_quantity:,} 张图片:\n')

    # annotations_tally 记录了每张图片内的标注数量，以及图片名字。
    annotations_tally = []
    max_annotations_in_one_image = pd.DataFrame({})
    counter_annotation = 0
    # showed_up_category_in_all_images 是在被统计的图片中，所有出现过的类别编号列表。
    showed_up_category_in_all_images = []

    for i, each in enumerate(image_path.iterdir()):
        # 从设定的第 start_image 张图片开始进行统计。
        if i >= start_image:
            image_name_int = int(each.stem)  # 去掉名字前面的多个数字 0。
            image_name = str(image_name_int)  # 需要将图片名转换为字符串。
            # annotations_in_one_image 是一个列表，代表图片里的所有标注内容。
            annotations_in_one_image = annotations_dict.get(image_name, [])
            # annotation_tally 是一个图片内的标注数量。
            annotation_tally = len(annotations_in_one_image)
            annotations_tally.append([annotation_tally, image_name_int])

            # 如果该图片内有标注，则统计其出现的类别，以及对应的数量。
            if annotations_in_one_image:
                # category_in_one_img 是一个图片内所有出现过的类别。
                category_in_one_img = []
                for each_annotation in annotations_in_one_image:
                    category_id_in_coco = each_annotation[0]
                    category_in_one_img.append(category_id_in_coco)

                # category_in_one_img_tally 是一个统计量，统计了所有出现的
                # 类别编号及其数量。
                category_in_one_img_tally = Counter(category_in_one_img)
                # max_annotations_each_image 是一张图片内，标注数量最高的类别编号
                # 及其数量。
                max_annotations_each_image = (
                    category_in_one_img_tally.most_common(1))

                id_in_coco = max_annotations_each_image[0][0]
                annotations_quantity = max_annotations_each_image[0][1]
                max_annotations_in_one_image.loc[
                    counter_annotation, 'category_id_in_coco'] = id_in_coco
                max_annotations_in_one_image.loc[
                    counter_annotation,
                    'annotations_quantity'] = annotations_quantity
                max_annotations_in_one_image.loc[
                    counter_annotation, 'image_name'] = image_name_int

                counter_annotation += 1

                # showed_up_category_in_one_img 是去掉了重复出现的类别。
                showed_up_category_in_one_img = set(category_in_one_img)
                showed_up_category_in_all_images += list(
                    showed_up_category_in_one_img)

            if i + 1 == start_image + images_quantity:
                break

    # annotations_array 形状为 (x, 2)，x 为统计图片的数量。
    annotations_array = np.asarray(annotations_tally)
    to_sort_annotations = annotations_array[:, 0]
    sorted_indexes = np.argsort(to_sort_annotations)
    # sorted_indexes 是排列后的索引，按照标注数量，从大到小排列。
    sorted_indexes = np.flip(sorted_indexes)
    # 把 annotations_array 按照标注数量，从大到小排列。
    annotations_array = annotations_array[sorted_indexes]
    print(f'一张图片内的最大标注数量:  {annotations_array[0, 0]}，  '
          f'该图片的名字编号为： {str(annotations_array[0, 1])}')

    showed_up_category_in_all_images_tally = Counter(
        showed_up_category_in_all_images)
    # max_showed_up_category_in_all_images 是所有被统计图片中，出现次数最多的类别。
    max_showed_up_category_in_all_images = (
        showed_up_category_in_all_images_tally.most_common(1))
    # max_showed_up_id_in_coco 是在 COCO 数据集中的类别 ID。
    max_showed_up_id_in_coco = max_showed_up_category_in_all_images[0][0]
    max_showed_up_times = max_showed_up_category_in_all_images[0][1]
    max_showed_up_row = FULL_CATEGORIES[
        FULL_CATEGORIES['id_in_coco'].isin([max_showed_up_id_in_coco])]
    # max_showed_up_name 是在 COCO 数据集中的类别名称。
    max_showed_up_name = FULL_CATEGORIES.at[max_showed_up_row.index[0], 'name']
    print(f'出现次数最多的类别编号:  {max_showed_up_id_in_coco}，  '
          f'该类别的名称为:  {max_showed_up_name}，  '
          f'该类别共出现在 {max_showed_up_times} 张图片中。')

    display_dataframe = max_annotations_in_one_image.set_index('image_name')
    display_dataframe = display_dataframe.sort_values(
        by='annotations_quantity', ascending=False)
    print(f'\n在每一张图片中，标注数量最多的类别记录如下（按照标注数量从大到小排列）:\n')
    display(display_dataframe)


if __name__ == '__main__':
    # 这部分代码，可以比较多线程和单线程的运行速度。
    images_quantity_list = [1000, 2000, 4_000, 8_000, 16_000,
                            32_000, 64_000, 120_000]

    group_images_threshold = 10_000  # 多进程的阈值，图片大于该阈值时，进行多进程并发。
    # 时间记录结果：4_000, 7.8;   8_000, 12.1;    10_000, 13.8;   12_000, 13.7;
    # 15_000, 13.7;     20_000, 12.8

    # 使用单个进程，统计 COCO 数据集的数据。
    single_proc_durations = []
    for images_quantity in images_quantity_list:
        tic = time.perf_counter()

        coco_statistics(
            datatype='train', start_image=0, images_quantity=images_quantity)

        toc = time.perf_counter()
        duration = round((toc - tic), 1)
        single_proc_durations.append(duration)

    # 使用多进程并发，统计 COCO 数据集的数据。
    multi_proc_durations = []
    for images_quantity in images_quantity_list:
        tic = time.perf_counter()

        coco_statistics_multi_processing(
            datatype='train', start_image=0, images_quantity=images_quantity,
            multi_processing_threshold=group_images_threshold)

        toc = time.perf_counter()
        duration = round((toc - tic), 1)
        multi_proc_durations.append(duration)

    print('\n' + '=' * 20 + ' Performance comparison ' + '=' * 20)
    print(f'group_images_threshold: {group_images_threshold:,}\n')
    duration_statistics = pd.DataFrame()
    for i, results in enumerate(zip(
            images_quantity_list, single_proc_durations, multi_proc_durations)):

        images_quantity, single_proc_duration, multi_proc_duration = results

        duration_statistics.loc[i, 'images_quantity'] = images_quantity
        duration_statistics.loc[
            i, 'single_process(seconds)'] = single_proc_duration
        duration_statistics.loc[
            i, 'multi_processes(seconds)'] = multi_proc_duration
        duration_statistics.loc[
            i, 'faster'] = round(single_proc_duration / multi_proc_duration, 1)

    duration_statistics = duration_statistics.set_index('images_quantity')

    display(duration_statistics)
