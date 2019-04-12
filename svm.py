#导入交叉验证库
from sklearn.model_selection import train_test_split#替换sklearn.cross_validation
#导入SVM分类算法库
from sklearn import svm
#生成预测结果准确率的混淆矩阵
from sklearn import metrics
import ReadData as rd
#数据文件夹
data_dir = "./data_mc_allinone/"
fpaths, datas, labels = rd.read_data(data_dir)

Y=labels
n_samples = len(datas)
X = datas.reshape((n_samples, 160*160*3))
#随机抽取生成训练集和测试集，其中训练集的比例为60%，测试集40%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 生成SVM分类模型
clf = svm.SVC(gamma=0.001)
# 使用训练集对svm分类模型进行训练
clf.fit(X_train, y_train)
#使用测试集衡量分类模型准确率
clf.score(X_test, y_test)
#对测试集数据进行预测
predicted=clf.predict(X_test)
#查看前20个测试集的预测结果
#predicted[:20]
#查看测试集中的真实结果
expected=y_test
#查看测试集中前20个真实结果
#expected[:20]
#生成准确率的混淆矩阵(Confusion matrix)
my_confusion=metrics.confusion_matrix(expected, predicted)

#测试集合的精度
sum = 0
for i in range(my_confusion.shape[0]):
    for j in range(i,my_confusion.shape[1]):
        if i==j:
            sum += my_confusion[i][j]
print('SVM acc rate is %.2f%%' % float(sum*100/len(y_test)))
print(my_confusion);

