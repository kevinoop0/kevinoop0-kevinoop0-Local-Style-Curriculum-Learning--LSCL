import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

def ConvBlock(c_in, c_out):
    "conv->bn->relu"
    func = [
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]
    return func


class InterpolateModule(nn.Module):
	def __init__(self, scale_factor=2):
		super().__init__()
		self.scale_factor = scale_factor

	def forward(self, x):
		return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)


class Unet(nn.Module):
    "Unet 2d"

    def __init__(self):
        super(Unet, self).__init__()

        self.conv0 = nn.Sequential(*ConvBlock(1, 64))
        self.conv0_1 = nn.Sequential(*ConvBlock(64, 64))
        self.conv1 = nn.Sequential(*ConvBlock(64, 128))
        self.conv1_1 = nn.Sequential(*ConvBlock(128, 128))
        for p in self.parameters():
            p.requires_grad=False
        self.conv2 = nn.Sequential(*ConvBlock(128, 256))
        self.conv2_1 = nn.Sequential(*ConvBlock(256, 256))
        self.conv3 = nn.Sequential(*ConvBlock(256, 512))
        self.conv3_1 = nn.Sequential(*ConvBlock(512, 512))
        self.conv4 = nn.Sequential(*ConvBlock(512, 512))
        self.conv4_1 = nn.Sequential(*ConvBlock(512, 512))

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upsample = InterpolateModule()

        self.deconv3 = nn.Sequential(*ConvBlock(1024, 512))
        self.deconv3_1 = nn.Sequential(*ConvBlock(512, 512))
        self.deconv2 = nn.Sequential(*ConvBlock(768, 256))  # 512 + 256
        self.deconv2_1 = nn.Sequential(*ConvBlock(256, 256))
        self.deconv1 = nn.Sequential(*ConvBlock(384, 128))  # 256 + 128
        self.deconv1_1 = nn.Sequential(*ConvBlock(128, 128))
        self.deconv0 = nn.Sequential(*ConvBlock(192, 64))  # 128 + 64
        self.deconv0_1 = nn.Sequential(*ConvBlock(64, 64))

        self.semantic_seg_conv = nn.Conv2d(64, 4, kernel_size=1)

        self.unet_encoder = [
            self.conv0, self.conv0_1, self.conv1, self.conv1_1, 
            self.conv2, self.conv2_1, self.conv3, self.conv3_1, self.conv4, self.conv4_1
        ]

        self.unet_decoder = [
            self.deconv3, self.deconv3_1, self.deconv2, self.deconv2_1,
            self.deconv1, self.deconv1_1, self.deconv0, self.deconv0_1
        ]

    def encode(self, x, states, level):
        assert level in {i for i in range(1, 11)}
        x = self.unet_encoder[level-1](x)
        states[f'enc_{level}'] = x
        if level in [2, 4, 6, 8]:
            x = self.maxpool(x)

        return x

    def decode(self, x, states, level):
        index = {1: 8, 3: 6, 5: 4, 7: 2}
        assert level in {i for i in range(1, 9)}
        if level in [1, 3, 5, 7]:
            x = self.upsample(x)
            x = torch.cat((x, states[f'enc_{index[level]}']), dim=1)
        x = self.unet_decoder[level-1](x)

        return x

    def forward(self, x):
        states = {}
        for level in range(1, 11):
            x = self.encode(x, states, level)
        for level in range(1, 9):
            x = self.decode(x, states, level)
        x = self.semantic_seg_conv(x)
        return x


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=1, output_ch=4):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        for p in self.parameters():
            p.requires_grad=False  
            
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(
            F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(
            F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(
            F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(
            filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()
    def first_layer(self, x):
        e1 = self.Conv1(x)
        return e1
    
    def other_layer(self, e1):
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out

    def forward(self, x, flag=False):
        if flag:
            out = self.other_layer(x)
        else:
            e1 = self.first_layer(x)
            out = self.other_layer(e1)

        return out


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class conv_block(nn.Module):
    """
    Convolution Block 
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
