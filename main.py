#!/usr/bin/env python
from __future__ import print_function

import torch
from torch import nn
import csdataset
from torchvision import transforms

from torchvision import models
import torchvision
from torch.autograd import Variable
from torch.utils import model_zoo

from tensorboardX import SummaryWriter
import torch.optim
import torch.nn.functional as F
import platform
import argparse

from PIL import Image


import time

class Writer:
    def __init__(self, data):
        self.writer = None
        self.data = data

    def init(self):
        if self.writer is None:
            self.writer = SummaryWriter()

        return self.writer



    def add_scalar(self, *args, **kwargs):
        self.init()
        return self.writer.add_scalar(*args, global_step=self.data.iter+1, **kwargs)

    def add_image(self, *args, **kwargs):
        self.init()
        return self.writer.add_image(*args, global_step=self.data.iter+1, **kwargs)




class SysConfig:
    def __init__(self):

        self.hostname = platform.node()

        self.style_path = None
        self.content_path = None

        if self.hostname == "Laptop":
            self.style_path = "C:/Users/Alex/Desktop/train_1"
            self.content_path = "C:/Users/Alex/Desktop/train_1"

        elif self.hostname == "LAPTOP-IVJKIKR4":
            self.style_path = "C:/Users/kroth/Downloads/val2017/"
            self.content_path = "C:/Users/kroth/Downloads/val2017/"

        elif self.hostname == "alexk-pc":
            self.style_path = "/mnt/ssd_data2/alexk/style-transfer/datasets/wikiart/train_1/"
            self.content_path = "/mnt/ssd_data2/alexk/style-transfer/datasets/coco/train2014/"
        else:
            raise Exception("Unknown host {}".format(self.hostname))


class VGGencoder(models.vgg.VGG):
    def __init__(self, fix_until):
        super(VGGencoder, self).__init__(models.vgg.make_layers(models.vgg.cfg['E']))
        self.load_state_dict(model_zoo.load_url(models.vgg.model_urls['vgg19']))
        self.named_layers = None

        self.fix_first_layers(fix_until)

    def fix_first_layers(self, until):
        mode = 0

        for layer_name, layer in zip(self.get_name_layers(), self.features.children()):
            if layer_name == until:
                mode = 1

            # print("Fixing " + layer_name)

            if mode == 0:
                for param in layer.parameters():
                    param.requires_grad = False
                    pass

            elif mode == 1:
                pass
                # for param in layer.parameters():
                #     param.data.normal_(0.0, 1.0)

        if mode == 0 and until is not None:
                raise Exception("Fixed all layers")

    def get_name_layers(self):
        if self.named_layers is None:

            self.named_layers = []

            counter_layers = 0
            counter_units = 1

            for module in self.features.children():
                cur_name = ""
                if isinstance(module, nn.Conv2d):
                    counter_layers += 1
                    prefix = "conv"
                elif isinstance(module, nn.ReLU):
                    prefix = "relu"
                elif isinstance(module, nn.MaxPool2d):

                    counter_units += 1
                    counter_layers = 0
                    prefix = "pool"
                else:
                    cur_name = "unknown"

                if cur_name != "unknown":
                    cur_name = "{}{}_{}".format(prefix, counter_units, counter_layers)


                self.named_layers.append(cur_name)

        return self.named_layers

    def forward(self, x):
        for layer_name, layer in zip(self.get_name_layers(), self.features.children()):

            x = layer(x)
            if layer_name == "relu4_1":
                break
        # x =self.features(x)

        return x



class VGGloss(models.vgg.VGG):
    def __init__(self):
        super(VGGloss, self).__init__(models.vgg.make_layers(models.vgg.cfg['E']))
        self.layer_names = []

        for param in self.parameters():
            param.requires_grad = False

    def reuse(self, encoder):
        self.layer_names = encoder.get_name_layers()

        my_params = {name: param for name, param in self.named_parameters()}

        for name, param in encoder.named_parameters():
            my_params[name].data = param.data




    def forward(self, x):
        #
        for layer_name, layer in zip(self.layer_names, self.features.children()):
            # print(layer_name)
            x = layer(x)
            if layer_name == "relu4_1":
                break

        # x = self.features(x)

        return x

    def run(self, x, return_names):

        nret = len(return_names)
        ret = [None] * nret


        for layer_name, layer in zip(self.layer_names, self.features.children()):


            x = layer(x)

            for idx, lname in enumerate(return_names):
                if lname == layer_name:
                    ret[idx] = x
                    nret -= 1

            if nret <= 0:
                break

        return ret
class Decoder(nn.Module):

    def get_conv(self, lin, lout, activation=True):
        f = [torch.nn.ReflectionPad2d(1),
            nn.Conv2d(lin, lout, 3)]

        if activation:
            f = f + [nn.ReLU()]


        return nn.Sequential(*f)

    def __init__(self):
        super(Decoder, self).__init__()
        # pseudo version
#        self.conv1 = nn.Conv2d(512,3,3,padding=1)
#        self.upsample = nn.Upsample(scale_factor=2**5)

        self.conv1uniform = nn.Sequential(
            *[self.get_conv(512, 512) for _ in range(3)]
        )

        self.conv2uniform = nn.Sequential(
           * ([self.get_conv(512, 512) for _ in range(0)] +
           [self.get_conv(512, 256)])
        )

        self.conv3uniform = nn.Sequential(
           *([self.get_conv(256, 256) for _ in range(3)] +
           [self.get_conv(256, 128)])
        )

        self.conv4uniform =self.get_conv(128, 128)
        self.conv5uniform = self.get_conv(64, 64)

        self.conv3 =self.get_conv(128, 64)
        self.conv4 =self.get_conv(64, 3,activation=False)
        #self.conv5 = self.get_conv(16, 3)

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # layer 1 maintain the same number of channels (512)
        # x = self.upsample(x)
        #
        # x = self.conv1uniform(x)

        # layer 2 decrease the number of channels from 521 to 256
        # x = self.upsample(x)

        x = self.conv2uniform(x)

        # layer 3 decrease the number of channels from 256 to 128
        x = self.upsample(x)


        x = self.conv3uniform(x)
        # layer 4 decrease number of channels from 128 to 64
        x = self.conv3(self.conv4uniform(self.upsample(x)))
        # layer 5 decrease number of channels from 64 to 16
        x =self.conv4(self.conv5uniform(self.upsample(x)))
        # last convolution to smooth and decrease channels from 16 to 3
        #x = self.conv5(x)
        return x


class NormCalc(nn.Module):
    def __init__(self):
        super(NormCalc, self).__init__()

    def forward(self, x):
        size = x.size()
        x_view = x.view(size[0], size[1], -1)

        avg = x_view.mean(2, keepdim=True)

        std = x_view.std(2, keepdim=True)

        std = std + 1e-5

        return avg.unsqueeze(-1), std.unsqueeze(-1)


class Normalizer(nn.Module):
    def __init__(self):
        super(Normalizer, self).__init__()
        self.normcalc = NormCalc()

    def forward(self, x):
        avg, std = self.normcalc(x)
        x = (x - avg) / std
        return x


class StatisticLoss(nn.Module):
    def __init__(self):
        super(StatisticLoss, self).__init__()
        self.normcalc = NormCalc()


    def forward(self, output, target):
        output_avg, output_std = self.normcalc(output)
        target_avg, target_std = self.normcalc(target)
        loss = F.mse_loss(output_avg, target_avg, size_average=False) + \
               F.mse_loss(output_std, target_std, size_average=False)

        return loss


class StyleLoss(nn.Module):
    def __init__(self, vgg_loss):
        super(StyleLoss, self).__init__()
        self.vgg_loss = vgg_loss
        self.layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1"]
        self.statistical_loss = StatisticLoss()

    def forward(self, output, target):
        output_features = self.vgg_loss.run(output, self.layers)
        target_features = self.vgg_loss.run(target, self.layers)

        total = 0

        for of, tf in zip(output_features, target_features):
            total += self.statistical_loss(of, tf)

        return total

class ContentLoss(nn.Module):
    def __init__(self, vgg_loss):
        super(ContentLoss, self).__init__()
        self.vgg_loss = vgg_loss


    def forward(self, output, target_vgg):
        output_vgg = self.vgg_loss.forward(output)

        return F.mse_loss(output_vgg, target_vgg)



class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
        self.normcalc = NormCalc()
        self.normalizer = Normalizer()

    def forward(self, content, style):
        # todo support multiple styles and mixing

        avg, std = self.normcalc(style)
        content = self.normalizer(content)

        x = (content* std) +avg

        return x

class ReverseNormalization:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self,  x):
        ret = []

        num_layers = int(x.shape[1])

        for idx in range(num_layers):
            ret.append(x[:,idx, :,:]*self.std[idx] + self.mean[idx])

        return torch.stack(ret, 1)

class Data:
    epoch = 0
    iter = 0
    ver = 0 # version

class StyleTransfer:
    vgg_mean = [0.485, 0.456, 0.406]
    vgg_std = [0.229, 0.224, 0.225]




    class RunConfig:
        save_loss_every = 10
        save_image_every = 10000
        save_model_every = 1000

    def __init__(self, args, gpu=True):

        self.args = args

        self.sysconfig = SysConfig()
        self.runconfig = StyleTransfer.RunConfig()

        self.use_gpu = torch.cuda.is_available() and gpu



        self.data = Data()

        self.max_epoch = 1000000

        self.encoder = self.to_gpu(VGGencoder("relu4_1"))
        self.decoder = self.to_gpu(Decoder())

        self.batch_size = 4

        self.adain = self.to_gpu(AdaIN())

        params_list = [param for param in self.encoder.parameters() if param.requires_grad] + list(self.decoder.parameters())
        params_list = self.decoder.parameters()
        self.optimizer = torch.optim.Adam(params_list, lr=0.0001)
        self.optimizer_type = "Adam"

        self.vgg_loss = self.to_gpu(VGGloss())
        self.vgg_loss.reuse(self.encoder)

        self.style_loss = self.to_gpu(StyleLoss(self.vgg_loss))
        self.content_loss = self.to_gpu(ContentLoss(self.vgg_loss))

        self.writer = Writer(self.data)

        self.to_screen_space = ReverseNormalization(self.vgg_mean, self.vgg_std)

    def load_state(self, name):
        load_obj = torch.load(name, map_location=('gpu' if self.use_gpu else 'cpu'))

        self.data = load_obj["Data"]
        self.writer.data = self.data

        networks = load_obj["Networks"]

        self.encoder.load_state_dict(networks['Encoder'])
        self.decoder.load_state_dict(networks['Decoder'])
        self.vgg_loss.reuse(self.encoder)

        if load_obj['Optimizer'][0] == self.optimizer_type:
            self.optimizer.load_state_dict(load_obj['Optimizer'][1])

    def save_state(self, name=None):
        if name is None:
            name = self.default_save_path

        save_obj = {"Data": self.data,
                    "Networks": {
                        "Encoder": self.encoder.state_dict(),
                        "Decoder": self.decoder.state_dict()
                        },
                    "Optimizer": [self.optimizer_type, self.optimizer.state_dict()]
                    }

        torch.save(save_obj, name)

    def to_gpu(self,x):
        if self.use_gpu:
            x = x.cuda()
            #
            # if isinstance(x, nn.Module):
            #     x = nn.DataParallel(x)

        return x

    def to_variable(self, x, requires_grad=True):
        x = self.to_gpu(x)
        x = Variable(x, requires_grad=requires_grad)

        return x

    def train(self):


        while self.data.epoch < self.max_epoch:

            print("[", end='')

            epoch_start_time = time.time()

            for idx, (image_content, image_style) in enumerate(self.dataloader):
                # print(image_content.type())
                # print(image_style.type())


                image_content = self.to_variable(image_content)
                image_style = self.to_variable(image_style)

                vgg_content = self.encoder(image_content)
                vgg_style = self.encoder(image_style)

                vgg_content_with_stlye = self.adain(vgg_content, vgg_style)
                # vgg_content_with_stlye = vgg_content

                stylized_content = self.decoder(vgg_content_with_stlye)

                style_loss = self.style_loss(stylized_content, image_style) * .01 / self.batch_size
                content_loss = self.content_loss(stylized_content, vgg_content_with_stlye)
                total_loss = style_loss + content_loss

#                total_loss = F.mse_loss(stylized_content,image_content )

                if (self.data.iter + 1) % self.runconfig.save_loss_every == 0:
                    self.writer.add_scalar('style_loss', style_loss)
                    self.writer.add_scalar('content_loss', content_loss)
                    self.writer.add_scalar('total_loss', total_loss)

                if (self.data.iter + 1) % self.runconfig.save_image_every == 0:
                    self.writer.add_image('image', torchvision.utils.make_grid(torch.cat(
                        [
                            self.to_screen_space(image_content.data),
                            self.to_screen_space(image_style.data),
                            self.to_screen_space(stylized_content.data).clamp(0,1),
                        ], 0
                    ), nrow=self.batch_size, padding=0))


                self.optimizer.zero_grad()
                self.encoder.zero_grad()
                self.content_loss.zero_grad()
                self.style_loss.zero_grad()
                self.vgg_loss.zero_grad()

                total_loss.backward()
                self.optimizer.step()

                if (self.data.iter + 1) % self.runconfig.save_model_every == 0:
                    self.save_state(self.args.model_out_name)
                    print("|", end='')

                print("=", end='')

                self.data.iter += 1

            print("]")

            print("Epoch {} is complete. Time: {} seconds".format(self.data.epoch,  time.time() - epoch_start_time))

            self.data.epoch +=1


    def create_train_dataset(self):
        crop_size = 256



        trans = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            lambda x: x.clamp(0,1),
            transforms.Normalize(mean=self.vgg_mean,
                                 std=self.vgg_std)
                ])

        cdataset = csdataset.CSDataset([crop_size,crop_size], self.sysconfig.content_path,
                                     self.sysconfig.style_path,
                                     trans, trans)


        self.dataloader = torch.utils.data.DataLoader(cdataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=0)

    
    def apply(self, style_path, content_path):
        # load two images
        style_img = image_loader(style_path)
        content_img = image_loader(content_path)
        
        # run through the network
        vgg_content = self.encoder(content_img)
        vgg_style = self.encoder(style_img)
 
        vgg_content_with_stlye = self.adain(vgg_content, vgg_style)
        vgg_content_with_stlye = vgg_content*.5 + vgg_content_with_stlye*.5

        stylized_content = self.decoder(vgg_content_with_stlye)
        
        # convert to tensor, remove 0 dimension, apply self.to_screen_space
        stylized_content = self.to_screen_space(stylized_content.data)
        stylized_content = stylized_content.squeeze(0)
        stylized_content = stylized_content.clamp(0,1)
        
        # convert to PIL and save
        stylized_content = unloader(stylized_content)
        stylized_content.save('stylelized_content.jpg')


    def run_apply(self):
        self.apply("style.jpg", "content.jpg")
            
loader = transforms.ToTensor()      # transform to tensor
unloader = transforms.ToPILImage()  # transform to PILImage

def image_loader(image_name):
    # load the image
    image = Image.open(image_name)
    # resize so divisible by 8 (cropped)
    width, height = image.size
    image = image.resize((width - width%8, height - height%8))
    
    # convert to tensor, Normalize and add batch dimension
    image = loader(image)
    image = Variable(image, requires_grad=False)
    
    f = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image = f(image)
    image = image.unsqueeze(0)
    return image


class Args:
    model_out_name = None
    model_in_name = None




def main():

    argp = Args
    parser = argparse.ArgumentParser(description='Style transfer.')
    parser.add_argument('--modelout', type=str, nargs=1,
                        help='File name for model output file',
                        default=["model.pth"])

    parser.add_argument('--modelin', type=str, nargs=1,
                        help='File name for model input file',
                        default=[None])
    
    parser.add_argument("--apply", action='store_true')

    args = parser.parse_args()

    argp.model_out_name = args.modelout[0]
    argp.model_in_name = args.modelin[0]

    st = StyleTransfer(gpu=True, args=argp)


    argp.model_in_name  = "../style_transfer/model.pth"
    if argp.model_in_name is not None:
        st.load_state(argp.model_in_name)

    st.create_train_dataset()
    
    if args.apply or True:
        st.run_apply()
    else:
        st.train()

if __name__ == "__main__":
    main()