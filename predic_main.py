import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def get_timestamp():
    """
    获取当前时间戳字符串
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class MLModel:
    def __init__(self):
        self.clf = None
        self.accuracy = None

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, model_name='Decision Tree'):
        print(f'{get_timestamp()} 开始训练{model_name}模型...')
        if model_name == 'Decision Tree':
            self.clf = DecisionTreeClassifier()
        elif model_name == 'KNN':
            self.clf = KNeighborsClassifier()
        elif model_name == 'Naive Bayes':
            self.clf = GaussianNB()
        elif model_name == 'SVM':
            self.clf = SVC()
        elif model_name == 'LogisticRegression':
            self.clf = LogisticRegression(max_iter=10000)
        elif model_name == 'RandomForestClassifier':
            self.clf = RandomForestClassifier()

        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        self.accuracy = accuracy_score(y_test, y_pred)
        print(f'{get_timestamp()} {model_name}预测准确率：{self.accuracy:.2%}')

        return self.accuracy

class IncomePredictionModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.label = None
        self.vectorized_data = None
        self.vectorized_label = None
        self.ml_model = MLModel()

    def read_file(self, skip_header=True):
        """
        读取CSV文件的内容。
        参数:
            skip_header (bool): 是否跳过表头数据，默认为True。
        """
        print(f'{get_timestamp()} 读取原始数据集文件: {self.file_path}')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            if skip_header:
                f.readline()  # 跳过表头数据
            reader = csv.reader(f)
            self.data = [row for row in reader]

    def split_data(self):
        """
        将数据分割为标签和数据。
        """
        print(f'{get_timestamp()} 开始分割数据...')
        data = [[col.strip() for col in row] for row in self.data]
        self.label = np.array([row[-1] for row in data])
        self.data = np.array([row[:-1] for row in data])

    def vectorize_data(self, onehot_cols, continuous_cols, exclude_cols=None):
        """
        使用scikit-learn将数据进行one-hot编码和标准化处理。
        """
        print(f'{get_timestamp()} 开始向量化数据...')
        data = self.data
        if exclude_cols:
            data = np.delete(data, exclude_cols, axis=1)
            onehot_cols = [col - sum(col > exc for exc in exclude_cols) for col in onehot_cols if col not in exclude_cols]
            continuous_cols = [col - sum(col > exc for exc in exclude_cols) for col in continuous_cols if col not in exclude_cols]
        else:
            onehot_cols = onehot_cols[:]
            continuous_cols = continuous_cols[:]

        onehot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_data = onehot_encoder.fit_transform(data[:, onehot_cols])

        scaler = StandardScaler()
        continuous_data = scaler.fit_transform(data[:, continuous_cols])

        self.vectorized_data = np.hstack((one_hot_data, continuous_data))

    def binarize_labels(self):
        """
        将标签列的数据从<=50K和>50K转换为二进制标签(0和1)。
        """
        print(f'{get_timestamp()} 开始二值化标签...')
        lb = LabelBinarizer()
        self.vectorized_label = lb.fit_transform(self.label).ravel()

    def train_and_evaluate_models(self):
        X_train, X_test, y_train, y_test = train_test_split(self.vectorized_data, self.vectorized_label, test_size=0.2)
        model_names = ['Decision Tree', 'KNN', 'Naive Bayes', 'SVM', 'LogisticRegression', 'RandomForestClassifier']
        accuracies = []

        for model_name in model_names:
            accuracies.append(self.ml_model.train_and_evaluate(X_train, X_test, y_train, y_test, model_name))

        return model_names, accuracies


def plot_accuracy_comparison(model_names, accuracies):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35

    # 绘制第一组柱状图
    bars = ax.bar(x - width / 2, accuracies, width, label='accuracies')

    # 为第一组柱状图添加数据标签
    for i, v in enumerate(accuracies):
        ax.text(x[i] - width / 2, v, f"{v:.2%}", ha='center', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Accuracy')
    ax.set_title('Different Models Accuracy Comparison')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # 实例化预测模型对象并执行
    model = IncomePredictionModel('./成人收入预测数据集.csv')
    model.read_file()
    model.split_data()

    # 设置onehot列为[1, 3, 5, 6, 7, 8, 9, 13]，连续列为[0, 2, 4, 10, 11, 12]，排除列为[2, 13]
    onehot_cols = [1, 3, 5, 6, 7, 8, 9, 13]
    continuous_cols = [0, 2, 4, 10, 11, 12]
    exclude_cols = [2, 13]
    model.vectorize_data(onehot_cols, continuous_cols, exclude_cols)
    model.binarize_labels()
    model_names, accuracies = model.train_and_evaluate_models()

    plot_accuracy_comparison(model_names, accuracies)
