import os

test_filepath = "data/people/test"    # 测试的图片的目录


for i in os.listdir(test_filepath):
    print("{}.jepg".format(".".join(i.split(".")[:-1])))
    # img = io.imread(os.path.join(test_path, "{}.png".format(i[:-4])), as_gray=as_gray)