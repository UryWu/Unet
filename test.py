from model import *
from data import *
# this python file was wrote by UryWu，其实在main里面已经对test文件夹目标分割了，但我重新写test的这部分
# 是为了每次可以直接使用h5文件来目标分割
num_test_samples = 5  # 测试的图片数量
test_filepath = "data/people/test"    # 测试的图片的目录

model = unet(pretrained_weights='unet_people.h5')
# model = unet()
# model.load_weights(filepath='unet_membrane.h5')  # 使用keras的函数加载模型
# 实际上每次你训练完一个模型后，你可以直接运行这里测试，因为我每次训练的模型会重复保存，
# 一个模型的名字为unet_你的类别.h5，另一个模型的名字包括了训练准确率、训练损失率和迭代次数

testGene = testGenerator(test_filepath)
results = model.predict_generator(testGene, num_test_samples, verbose=1)
saveResult(test_filepath, results)

'''
UserWarning: data/people/test\3_predict.png is a low contrast image
warn('%s is a low contrast image' % fname)
出现这个警告：你的图像是低对比图像，目标分割不出来，估计要更多数据更多迭代来训练模型。警告在第几行你的第几个图片就预测为空白。

generator_output = next(output_generator)
StopIteration
出现这个错误：说明你的测试的图片数量没改对，翻上去改一下
'''
