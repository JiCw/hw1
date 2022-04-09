# coding=utf-8
import numpy as np
import struct
import os
import time
from tqdm import tqdm
from networks import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
import matplotlib.pyplot as plt
import json

MNIST_DIR = "./mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


class MNIST_MLP(object):
    def __init__(self, batch_size=1, input_size=784, hidden1=256, out_classes=10, lr=0.0001, max_epoch=30, decay=0.99, print_iter=20000, lambda_=0.5):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.lambda_ = lambda_
        self.decay = decay

    def load_mnist(self, file_dir, is_images=True):
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        if is_images:
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        return mat_data

    def load_data(self):
        print('Loading MNIST data from files...')
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)
        # self.test_data = np.concatenate((self.train_data, self.test_data), axis=0)

    def shuffle_data(self): # 随机抽取数据
        np.random.shuffle(self.train_data)

    def build_model(self):  # 建立网络结构
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1, self.lambda_)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.out_classes, self.lambda_)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2]

    def init_model(self):
        for layer in self.update_layer_list:
            layer.init_param()

    def load_model(self, param_dir):
        params = np.load(param_dir).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])

    def save_model(self, param_dir):
        print('Saving parameters to file ' + param_dir)
        self.params = {}
        self.params['w1'], self.params['b1'] = self.fc1.save_param()
        self.params['w2'], self.params['b2'] = self.fc2.save_param()
        np.save(param_dir, self.params)

    def forward(self, input):  # 神经网络的前向传播
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        prob = self.softmax.forward(h2)
        return prob

    def backward(self):  # 神经网络的反向传播
        dloss = self.softmax.backward()
        dh2 = self.fc2.backward(dloss)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def train(self):
        max_batch = self.train_data.shape[0] // self.batch_size
        train_loss, test_loss = [], []
        test_acc = []
        print('Start training...')
        for idx_epoch in tqdm(range(self.max_epoch)):
            self.lr *= pow(self.decay, idx_epoch)
            epoch_loss = []
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, -1]
                self.prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                epoch_loss.append(loss)
                self.backward()
                self.update(self.lr)
            train_loss.append(np.sum(epoch_loss)/len(epoch_loss))
            accuracy, test_epoch_loss = self.evaluate()
            test_loss.append(np.sum(test_epoch_loss)/len(test_epoch_loss))
            test_acc.append(accuracy)
            self.save_model('params')
        epochs = range(self.max_epoch)

        plt.figure(figsize=(8, 6))  # 定义图的大小
        plt.xlabel("epochs")  # X轴标签
        plt.ylabel("loss")  # Y轴坐标标签
        plt.plot(epochs, train_loss)
        plt.plot(epochs, test_loss)  # 绘制曲线图
        plt.legend(['training loss', 'test_loss'], loc='upper left')
        plt.savefig('./train&test_loss.jpg')

        plt.figure(figsize=(8, 6))  # 定义图的大小
        plt.xlabel("epochs")  # X轴标签
        plt.ylabel("acc")  # Y轴坐标标签
        plt.title("test accuracy")  # 曲线图的标题
        plt.plot(epochs, test_acc)  # 绘制曲线图
        plt.savefig('./test_acc.jpg')

    def evaluate(self):
        test_loss = []
        pred_results = np.zeros([self.test_data.shape[0]])
        start_time = time.time()
        for idx in range(self.test_data.shape[0]//self.batch_size):
            batch_images = self.test_data[idx*self.batch_size:(idx+1)*self.batch_size, :-1]
            batch_labels = self.train_data[idx * self.batch_size:(idx + 1) * self.batch_size, -1]
            self.prob = self.forward(batch_images)
            loss = self.softmax.get_loss(batch_labels)
            test_loss.append((loss))
            pred_labels = np.argmax(self.prob, axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print("Time for one epoch: %f" % (time.time()-start_time))
        print('Accuracy in test set: %f' % accuracy)
        return accuracy, test_loss



def build_mnist_mlp(param_dir='weight.npy'):
    h1, e, lambda_ = 200, 30, 4
    mlp = MNIST_MLP(hidden1=h1, max_epoch=e, lambda_=lambda_)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    start_time = time.time()
    mlp.train()
    print("All training and test time: %f" % (time.time()-start_time))
    return mlp

if __name__ == '__main__':
    mlp = build_mnist_mlp()
    
    