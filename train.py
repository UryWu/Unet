from model import *
from data import *

# 本工程来源：https://blog.csdn.net/py_yangh/article/details/83003972

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用编号为0的GPU，使用多个GPU则： = "0,1,2,3,4,5,6,7"
batch_size = 6  # 每次放入网络的图片数量
num_validation_samples = 12  # 验证集数量
num_train_samples = 24  # 训练集数量
epochs = 1000  # 迭代次数

data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
                     zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')
myGene = trainGenerator(batch_size, 'data/people/train', 'image', 'label', data_gen_args, save_to_dir=None)
print(myGene)
validation_generator = trainGenerator(batch_size, 'data/people/validation', 'image', 'label', data_gen_args,
                                      save_to_dir=None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_people.h5', monitor='loss', verbose=1, save_best_only=True)
# ModelCheckpoint:保存训练结果模型的设置，'unet_people.hdf5'是保存在当前目录下，名字为此。verbose: 详细信息模式，0 或者 1，
# ModelCheckpoint:如果设置save_weights_only=True，每次保存网络只保存权重


history = model.fit_generator(myGene, steps_per_epoch=num_train_samples // batch_size, epochs=epochs,
                              callbacks=[model_checkpoint],
                              validation_data=validation_generator,
                              validation_steps=num_validation_samples // batch_size)
# fit_generator: keras.callbacks.Callback 实例的列表。在训练时调用的一系列回调函数，这个就是用来保存训练结果的模型，steps_per_epoch每次迭代的优化器优化次数

val_acc, epoch = vitualization(history, 'people')  # 这里后面的这个字符串将写入训练结果可视化图片的名字中
# 训练过程的损失值、准确值可视化

model.save_weights('./unet_people_accur={0:.2f}_epoch={1}.h5'.format(val_acc, epoch))
# 保存模型，实际上上面已经保存了，但我这里保存的名字带有训练的信息

