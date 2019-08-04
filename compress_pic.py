
from PIL import Image
import os
from glob import glob

fpath = "./data/people/validation/image"  #有四个目录 "./data3/validation/1"  "./data3/train/0"  "./data3/train/1"
size = (512, 512)  # 定义要调整成为的尺寸（PIL会自动根据原始图片的长宽比来缩放适应设置的尺寸）
# glob.glob()用来进行模糊查询，增加参数recursive=True后可以使用**/来匹配所有子目录
files = glob(fpath + "**/*.jpg", recursive=True) + glob(fpath + "**/*.jpg", recursive=True)  # 如果输入的图是jpg,写jpg
total = len(files)  # 总文件数
cur = 1  # 当前文件序号
print("共有" + str(total) + "个文件，开始处理")
print("-----------------------------------")
for infile in files:
    try:
        # f, ext = os.path.splitext(infile) # 分离文件名和后缀
        print("进度:" + str(cur) + "/" + str(total) + "   " + infile)
        img = Image.open(infile)  # 打开图片文件
        if img.width > 100:
            img.thumbnail(size, Image.ANTIALIAS)  # 使用抗锯齿模式生成缩略图（压缩图片）
            img.save(infile, "jpeg")  # 保存成与原文件名一致的文件，会自动覆盖源文件，如果是jpg图写jpeg
        else:
            print(infile + "宽度小于1200px，无需处理，已忽略")
        cur = cur + 1

    except OSError:
        print(infile + "文件错误，忽略")
