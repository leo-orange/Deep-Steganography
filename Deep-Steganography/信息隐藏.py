import torch
from model import Hide, Reveal
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from ssim3 import SSIM,MS_SSIM
import cv2
import numpy as np
import math

#将图片进行裁剪并且转换为tensor形式
def transforms_img(img):
    img = Image.open(img) # 读取图片
    img = img.resize((256, 256))
    tensor = transforms.ToTensor()(img) # 将图片转化成tensor，
    # print(tensor.shape) # [3, 224, 224]
    tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
    # print(tensor.shape)# [1,3, 224, 224]
    tensor = tensor.cuda()
    return tensor
# 将模型进行导入
def moxing(hide):
    hide_net = Hide()
    hide_net.eval()

    hide_net.cuda()

    hide_net.load_state_dict(torch.load(hide))
    return hide_net
#模型指标损失函数计算
def norm(secret,cover):
    ssim=SSIM()
    loss=1-ssim(secret,cover)
    print(loss)

    # print(cover)
    # ssim_loss = pytorch_ssim.ssim(secret,cover)
    # ssim_loss = pytorch_ssim.SSIM()
    # print(ssim_loss(secret,cover))
    # print(ssim_loss)

    # criterion = nn.MSELoss()
    # loss_r = criterion(cover,output)
    # print("loss_r:" + str(loss_r.item()))

    # SSIM_h=1-pytorch_msssim.ssim(cover,output,nonnegative_ssim=False)
    # print("SSIM_h:"+str(SSIM_h.item()))

    # MS_SSIM_h=1-pytorch_msssim.ms_ssim(cover,output)
    # print("MS_SSIM_h:"+str(MS_SSIM_h.item()))

def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100

    return 10 * math.log10(255.0 ** 2 / mse)

if __name__ == '__main__':
    # MSE Loss SSIM MS_SSIM
    hide='D:/毕业设计/代码/ssim算法/checkpoint/epoch_1900_hide.pkl'
    hide_net=moxing(hide)

    secret_img='./result1/cover.png'
    cover_img='./result1/secret.png'

    secret=transforms_img(secret_img)
    cover=transforms_img(cover_img)

    output = hide_net(secret, cover)
    norm(cover,cover)


    save_image(secret.cpu().data[:4],fp='./result1/secret.png')
    save_image(cover.cpu().data[:4],fp='./result1/cover.png')
    save_image(output.cpu().data[:4],fp='./result1/output.png')
    gt = cv2.imread('./result1/cover.png')
    img = cv2.imread('./result1/output.png')
    print(psnr1(gt, img))

