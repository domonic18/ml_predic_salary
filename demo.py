import csv
import numpy as np

def read_file(file_path, skip_header=True):
  """
  读取CSV文件的内容。
  参数:
      file_path (str): CSV文件的路径。
      skip_header (bool): 是否跳过表头数据，默认为True。
  返回:
      list: 包含CSV文件内容的列表。
  """
  print(f'读取原始数据集文件: {file_path}')
  with open(file_path, 'r', encoding='utf-8') as f:
      if skip_header:
          # 跳过表头数据
          f.readline()
      reader = csv.reader(f)
      return [row for row in reader]  # 读取csv文件的内容


def split_data(data):
    """
    将数据分割为标签和数据。
    参数:
        data (list): 数据行的列表，第一个元素是标签。
    返回:
        numpy.ndarray: 标签数组。
        numpy.ndarray: 连接元素后的数据数组。
    """

    # 去除每个元素的前后空格
    data = [[col.strip() for col in row] for row in data]

    # 分离数据和标签
    n_label = np.array([row[-1] for row in data])
    n_data = np.array([row[:-1] for row in data])

    return n_label, n_data

from sklearn.preprocessing import OneHotEncoder, StandardScaler

def vectorize_data_with_sklearn(data, onehot_cols, continuous_cols, exclude_cols=None):
    """
    使用scikit-learn将给定的NumPy数组中的数据进行one-hot编码和标准化处理。

    参数:
    data (np.ndarray): 输入的NumPy数组
    onehot_cols (list): 需要进行one-hot编码的列索引
    continuous_cols (list): 不需要one-hot编码的连续量列索引
    exclude_cols (list, optional): 需要排除的列索引

    返回:
    np.ndarray: 经过one-hot编码和标准化的向量化数据
    """
    # 排除不需要处理的列
    if exclude_cols:
        data = np.delete(data, exclude_cols, axis=1)
        onehot_cols = [col - sum(col > exc for exc in exclude_cols) for col in onehot_cols if col not in exclude_cols]
        # 解释过程:
        # 1. 遍历 onehot_cols 中的每个索引 col
        # 2. 检查 col 是否在 exclude_cols 中
        #    - 如果在,计算 col 在 exclude_cols 中的位置 sum(col > exc for exc in exclude_cols)
        #    - 并从 col 中减去这个值,更新 col 的索引
        #    - 例如: col = 1, 在 exclude_cols 中的位置为 0, 则更新后 col = 1 - 0 = 1
        #    - ⭐ 这样可以确保 onehot_cols 中的索引能正确对应到数据的列
        # 3. 如果 col 不在 exclude_cols 中,则保留原始索引

        continuous_cols = [col - sum(col > exc for exc in exclude_cols) for col in continuous_cols if col not in exclude_cols]
        # 解释过程:
        # 1. 遍历 continuous_cols 中的每个索引 col
        # 2. 检查 col 是否在 exclude_cols 中
        #    - 如果在,计算 col 在 exclude_cols 中的位置 sum(col > exc for exc in exclude_cols)
        #    - 并从 col 中减去这个值,更新 col 的索引
        #    - 例如: col = 2, 在 exclude_cols 中的位置为 0, 则更新后 col = 2 - 0 = 2
        #    - ⭐ 这样可以确保 continuous_cols 中的索引能正确对应到数据的列
        # 3. 如果 col 不在 exclude_cols 中,则保留原始索引

    else:
        onehot_cols = onehot_cols[:]
        continuous_cols = continuous_cols[:]

    # 对离散量列进行one-hot编码
    onehot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_data = onehot_encoder.fit_transform(data[:, onehot_cols])

    # 对连续量列进行标准化
    scaler = StandardScaler()
    continuous_data = scaler.fit_transform(data[:, continuous_cols])

    # 将one-hot编码结果和标准化后的连续量列拼接起来
    final_data = np.hstack((one_hot_data, continuous_data))

    return final_data

import numpy as np
from sklearn.preprocessing import LabelBinarizer

def binarize_labels(labels):
    """
    将标签列的数据从<=50K和>50K转换为二进制标签(0和1)。
    参数:
        labels (np.ndarray): 输入的标签列
    返回:
        np.ndarray: 二进制标签
    """
    lb = LabelBinarizer()
    binary_labels = lb.fit_transform(labels)
    return binary_labels


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def decision_tree(label, data, test_size=0.2):
    """
    决策树模型的训练和评估。
    参数:
        train_data_file_path (str): 向量化数据的文件路径。
        test_size (float): 测试集的比例，默认为0.2。
    """
    print('开始加载训练数据...')

    # 训练集和测试集切分
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

    print('开始训练决策树模型...')
    # 数据预测
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 评估
    print('开始决策树预测...')
    accuracy = np.mean(y_pred == y_test)
    print(f'预测准确率：{accuracy}')


if __name__ == "__main__":
    # 读取文件
    listdata = read_file('./成人收入预测数据集.csv')
    # 对数据进行切分
    label, data = split_data(listdata)

    # 对数据进行切分
    onehot_cols = [1, 3, 5, 6, 7, 8, 9, 13]
    continuous_cols = [0, 2, 4, 10, 11, 12]

    # 排除列增加最后一列
    exclude_cols = [2, 13]
    vectorized_data = vectorize_data_with_sklearn(data, onehot_cols, continuous_cols, exclude_cols)

    vectorized_label = binarize_labels(label)

    decision_tree(vectorized_label, vectorized_data)
