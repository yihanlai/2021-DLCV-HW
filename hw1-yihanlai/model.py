import torch
import torch.nn as nn
import torchvision


class VGG16_FCN32(nn.Module):
    def __init__(self, num_classes):
        super(VGG16_FCN32, self).__init__()
        self.num_classes = num_classes

        # conv1-5
        self.vgg16 = torchvision.models.vgg16(pretrained=True).features
        # conv6-7
        self.fconv = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        self.score = nn.Conv2d(4096, self.num_classes, 1)

    def forward(self, x):
        vgg16 = self.vgg16(x)
        fconv = self.fconv(vgg16)
        score = self.score(fconv)

        return nn.functional.upsample_bilinear(score, x.size()[2:])


class VGG16_FCN8(nn.Module):
    def __init__(self, num_classes):
        super(VGG16_FCN8, self).__init__()
        self.num_classes = num_classes
        # conv1-5
        feats = list(torchvision.models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:10])
        self.feat3 = nn.Sequential(*feats[10:17])
        self.feat4 = nn.Sequential(*feats[17:24])
        self.feat5 = nn.Sequential(*feats[24:30])

        # conv6-7
        self.fconv = nn.Sequential(
                    nn.Conv2d(512, 4096, 7),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Conv2d(4096, 4096, 1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    )
        self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)
        self.score_fconv = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat3 = self.feat3(feats)
        feat4 = self.feat4(feat3)
        feat5 = self.feat5(feat4)
        fconv = self.fconv(feat5)

        score_feat3 = self.score_feat3(feat3)
        score_feat4 = self.score_feat4(feat4)
        score_fconv = self.score_fconv(fconv)

        score = nn.functional.upsample_bilinear(score_fconv, score_feat4.size()[2:])
        score += score_feat4
        score = nn.functional.upsample_bilinear(score, score_feat3.size()[2:])
        score += score_feat3

        return nn.functional.upsample_bilinear(score, x.size()[2:])



# Convolution block: two 3x3 convolution layers
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


# Encoder block: an conv_block and max pooling
class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)

        return x, p


# Decoder block: begins with a transpose conv, followed by a concatenation, next comes the conv_block
class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = conv_block(out_channels*2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        # Encoder
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        # Bottleneck
        self.trans = conv_block(512, 1024)

        # Decoder
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        
        #classifier
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bottleneck
        trans = self.trans(p4)

        # Decoder
        d1 = self.d1(trans, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # Classifier
        output = self.output(d4)

        return output
