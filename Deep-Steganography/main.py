import torch
from model import Hide, Reveal
from utils import DatasetFromFolder
import torch.nn.init as init
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from ssim3 import SSIM

# 初始化权重值
def init_weights(m):
    classname = m.__class__.__name__ #得到网络层的名字，如ConvTranspose2d
    if classname.find('Conv') != -1: #使用了find函数，如果不存在返回值为-1，所以让其不等于-1
        # kaiming高斯初始化
        # tensor是torch.Tensor变量，
        # a为Relu函数的负半轴斜率，
        # mode表示是让前向传播还是反向传播的输出的方差为1，
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # 初始化权重值
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 主函数入口
if __name__ == '__main__':
    '''
    将放置图片的文件夹进行导入，并对图片进行裁剪表明其大小为256
    将图片信息进行导入封装成对象的形式，封装成数据集的形式
    将文件夹中的前一半(400)张图片作为秘密图片、后一半(400)张图片作为载体图片
    一共400个数据集
    '''
    dataset = DatasetFromFolder('./data', crop_size=256)
    # 然后利用torch.utils.data.DataLoader将整个数据集分成多个批次。将400数据集以每组32个进行导出运算
    dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=4)
    # 生成模型实例
    hide_net = Hide()
    # 初始化权重
    '''
    apply函数将递归地搜索网络中的所有模块，并在每个模块上调用该函数。
    因此，您模型中的所有线性层都将使用这个调用进行初始化。
    '''
    hide_net.apply(init_weights)#递归的调用weights_init函数,遍历hide_net的submodule作为参数
    # 将解析网络进行导入
    reveal_net = Reveal()
    # 初始化权重
    reveal_net.apply(init_weights)

    # 将损失函数进行导入
    # criterion = nn.MSELoss()
    ssim = SSIM()

    # 将网络放在gpu上进行计算
    hide_net.cuda()
    reveal_net.cuda()
    # criterion.cuda()
    ssim.cuda()

    # 如果想要使用.cuda()方法来将model移到GPU中，一定要确保这一步在构造Optimizer之前。
    # 因为调用.cuda()之后，model里面的参数已经不是之前的参数了。
    # 优化算法，采用的是Adam优化的方法，学习速率为1e-3
    '''
    从优化器的作用出发，要使得优化器能够起作用，需要主要两个东西：
    1. 优化器需要知道当前的网络或者别的什么模型的参数空间，这也就是为什么在训练文件中，
    正式开始训练之前需要将网络的参数放到优化器里面，比如使用pytorch的话总会出现类似如下的代码：
    '''
    optim_h = optim.Adam(hide_net.parameters(), lr=1e-3)
    optim_r = optim.Adam(reveal_net.parameters(), lr=1e-3)
    # MultiStepLR能够控制学习率，milestones表示训练的epoch里程碑，gamma表示衰减因子
    schedulee_h = MultiStepLR(optim_h, milestones=[100, 1000])
    schedulee_r = MultiStepLR(optim_h, milestones=[100, 1000])

    # 进行2000次的循环和训练
    for epoch in range(2000):
        # 按照Pytorch的定义是用来更新优化器的学习率的，
        # 一般是按照epoch为单位进行更换，即多少个epoch后更换一次学习率
        schedulee_h.step()
        schedulee_r.step()
        # 现将隐藏网络的缺失值与显示网络的缺失值进行隐藏
        epoch_loss_h = 0.
        epoch_loss_r = 0.

        for i, (secret, cover) in enumerate(dataloader):
            # torch.Tensor是pytorch中训练时所采取的向量格式
            # 内部新建的张量是否放到了CUDA上
            secret = Variable(secret).cuda()
            cover = Variable(cover).cuda()
            # 梯度初始化为零
            # （因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
            optim_h.zero_grad()
            optim_r.zero_grad()
            # 利用隐藏网络进行信息隐藏，这里要将secret张量放在gpu上
            output = hide_net(secret, cover)

            # 利用批判网络对输出后图片与原来的载体图片进行缺失值比较
            # loss_h = criterion(output, cover)
            loss_h =1-ssim(output, cover)
            # 将损失网络模型中的结果取出
            epoch_loss_h += loss_h.item()

            reveal_secret = reveal_net(output)
            # 利用批判网络与解析后的秘密图片与原来的秘密图片图片进行缺失值比较
            # loss_r = criterion(reveal_secret, secret)
            loss_r =1-ssim(reveal_secret, secret)
            epoch_loss_r += loss_r.item()
            # 将损失网络模型中的结果取出

            # 计算整体缺失值
            loss = loss_h + 0.75 * loss_r

            print(loss)

            '''
            step这个函数使用的是参数空间(param_groups)中的grad,
            也就是当前参数空间对应的梯度，这也就解释了为什么optimzier使用之前需要zero清零一下，
            因为如果不清零，那么使用的这个grad就得同上一个mini-batch有关，这不是我们需要的结果。
            再回过头来看，我们知道optimizer更新参数空间需要基于反向梯度，
            因此，当调用optimizer.step()的时候应当是loss.backward()的时候
            '''
            # 自动求导函数
            loss.backward()
            # 更新隐藏网络模型
            optim_h.step()
            # 更新显示网络模型
            optim_r.step()
            # 将此次模型中的秘密图片、解密后的图片、载体图片以及加密图片进行输出
            if i == 3 and epoch % 20 == 0:
                save_image(torch.cat([secret.cpu().data[:4], reveal_secret.cpu().data[:4], cover.cpu().data[:4], output.cpu().data[:4]], dim=0), fp='./result/res_epoch_{}.png'.format(epoch), nrow=4)
        # 输出情况下模型的隐藏网络的缺失值和显示网络的缺失值
        print('epoch {0} hide loss: {1}'.format(epoch, epoch_loss_h))
        print('epoch {0} reveal loss: {1}'.format(epoch, epoch_loss_r))
        print('=======>>>'*5)

        # 将大于1000次循环之后每100次输出隐藏隐藏网络和解密网络的信息进行模型保存
        if epoch > 1000 and epoch % 100 == 0:
            torch.save(hide_net.state_dict(), './checkpoint/epoch_{}_hide.pkl'.format(epoch))
            torch.save(reveal_net.state_dict(), './checkpoint/epoch_{}_reveal.pkl'.format(epoch))






























