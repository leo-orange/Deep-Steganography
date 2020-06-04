import torch
from model import Hide, Reveal
from utils import DatasetFromFolder
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import cv2
import numpy as np
import math

def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100

    return 10 * math.log10(255.0 ** 2 / mse)
# 将模型进行导入
def hide_moxing(hide):
    hide_net = Hide()
    hide_net.eval()

    hide_net.cuda()

    hide_net.load_state_dict(torch.load(hide))
    return hide_net
def reveal_moxing(reveal):
    reveal_net = Reveal()
    reveal_net.eval()
    reveal_net.cuda()

    reveal_net.load_state_dict(torch.load(reveal))
    return reveal_net
if __name__ == '__main__':
    dataset = DatasetFromFolder('./datatext', crop_size=256)
    # 然后利用torch.utils.data.DataLoader将整个数据集分成多个批次。将400数据集以每组32个进行导出运算
    dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=4)
    hide = 'D:/毕业设计/代码/ssim算法/checkpoint/epoch_1900_hide.pkl'
    hide_net = hide_moxing(hide)
    reveal = 'D:/毕业设计/代码/ssim算法/checkpoint/epoch_1900_reveal.pkl'
    reveal_net = reveal_moxing(reveal)

    for i, (secret, cover) in enumerate(dataloader):
        secret = Variable(secret).cuda()
        cover = Variable(cover).cuda()

        output = hide_net(secret, cover)
        reveal_secret = reveal_net(output)

        save_image(secret.cpu().data[:4], fp='./result/secret.png')
        save_image(cover.cpu().data[:4], fp='./result/cover.png')
        save_image(reveal_secret.cpu().data[:4], fp='./result/reveal_secret.png')
        save_image(output.cpu().data[:4], fp='./result/output.png')
        gt = cv2.imread('./result/secret.png')
        img = cv2.imread('./result/reveal_secret.png')
        print(psnr1(gt, img))