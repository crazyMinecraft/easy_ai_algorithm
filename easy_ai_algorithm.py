import cv2
import numpy as np
from sklearn import svm

# 读取验证码图像
img = cv2.imread('captcha.png')

# 将图像转换成灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化处理
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 轮廓检测
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 特征提取，提取每个字符的高度、宽度、面积等特征
features = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    features.append([w, h, cv2.contourArea(contour)])

# 数据预处理，将特征转换成数字表示，并进行归一化处理
X = np.array(features, dtype=np.float32)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X = X.reshape(-1, 3)

# 标签数据，也就是验证码字符
y = ['A', 'B', 'C', 'D']

# 划分数据集，将数据集划分成训练集和测试集
train_X, train_y = X[:3], y[:3]
test_X, test_y = X[3:], y[3:]

# 训练模型
clf = svm.SVC()
clf.fit(train_X, train_y)

# 模型评估
score = clf.score(test_X, test_y)
print(score)

# 应用模型，使用模型对新的验证码进行识别
new_img = cv2.imread('new_captcha.png')
new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
_, new_binary = cv2.threshold(new_gray, 127, 255, cv2.THRESH_BINARY)
new_contours, _ = cv2.findContours(new_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
new_features = []
for new_contour in new_contours:
    new_x, new_y, new_w, new_h = cv2.boundingRect(new_contour)
    new_features.append([new_w, new_h, cv2.contourArea(new_contour)])

# 数据预处理，将新验证码的特征转换成数字表示，并进行归一化处理
new_X = np.array(new_features, dtype=np.float32)
new_X = (new_X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
new_X = new_X.reshape(-1, 3)

# 使用训练好的模型进行预测
predict_y = clf.predict(new_X)

# 输出预测结果
print(predict_y)
