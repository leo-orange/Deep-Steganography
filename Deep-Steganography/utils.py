import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

#根据图片路径打开图片
def load_img(filepath):
    img = Image.open(filepath)
    return img

'''
 torchvision.transforms是pytorch中的图像预处理包
 crop_size图片的大小
 RandomCrop：在一个随机的位置进行裁剪
'''
def input_transform(crop_size):
    return transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
    ])
class DatasetFromFolder(data.Dataset):
    # image_dir是图片的路径、crop_size是将图片裁剪的大小
    def __init__(self, image_dir, crop_size):
        # 将继承DatasetFromFolder类,并对类进行转换为self,使self可以调用类中的信息
        super(DatasetFromFolder, self).__init__()
        # 作为图像的预处理，设定图像的大小的裁剪形式
        self.input_transform = input_transform(crop_size)
        #将所有图片中的地址进行导入并封装成list集合['./datatext\\0002.png', './datatext\\0003.png']
        # 这种只适合一个文件夹内全是图片的，子文件夹内图片不会读取
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
        # 获取到秘密图像的地址信息['./datatext\\0002.png']
        self.secret_filenames = self.image_filenames[:len(self.image_filenames)//2]
        # 获取到覆盖图像的地址信息['./datatext\\0003.png']
        self.cover_filenames = self.image_filenames[len(self.image_filenames)//2:]
    # 其中__getitem__函数的作用是根据索引index遍历数据
    def __getitem__(self, index):
        # print("1")
        secret = load_img(self.secret_filenames[index])
        cover = load_img(self.cover_filenames[index])
        if self.input_transform:
            secret = self.input_transform(secret)
            cover = self.input_transform(cover)
        return secret, cover
    # __len__函数的作用是返回数据集的长度
    def __len__(self):
        # print(self.secret_filenames)
        return len(self.secret_filenames)
