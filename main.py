#!/usr/bin/env python
import torch
from torch import nn
import csdataset
from torchvision import transforms

from torchvision import models
from torch.autograd import Variable
from torch.utils import model_zoo
# from tensorboardX import SummaryWriter







class VGGencoder(models.vgg.VGG):
    def __init__(self):
        super(VGGencoder, self).__init__(models.vgg.make_layers(models.vgg.cfg['E']))
        self.load_state_dict(model_zoo.load_url(models.vgg.model_urls['vgg19']))
        self.named_layers = None

    def fix_first_layers(self, until):
        for layer_name, layer in zip(self.get_name_layers(), self.features.children()):
            if layer_name == until:
                break

            print("Fixing " + layer_name)
            for param in layer.parameters():
                param.requires_grad = False

        else:
            if until is not None:
                raise Exception("Ffixed all layers")

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

                print(cur_name)
                self.named_layers.append(cur_name)

        return self.named_layers

    def forward(self, x):
        x = self.features(x)
        return x



class VGGloss(models.vgg.VGG):
    def __init__(self):
        super(VGGloss, self).__init__(models.vgg.make_layers(models.vgg.cfg['E']))
        self.layer_names = []
        self.return_names = []

    def set_outputs(self, return_names):
        self.return_names = return_names


    def reuse(self, encoder):
        self.layer_names = encoder.get_name_layers()

        my_params = {name: param for name, param in self.parameters()}

        for name, param in encoder.named_parameters():
            my_params[name].data = param.data
            my_params[name].requires_grad = False



    def forward(self, x):

        nret = len(self.return_names)
        ret = [None] * nret


        for layer_name, layer in zip(self.networks.children()):


            x = layer(x)

            for idx, lname in enumerate(self.return_names):
                if lname == layer_name:
                    ret[idx] = x
                    nret -= 1

            if nret <= 0:
                break

        return ret
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        return x


def temp():
    vgg19 = models.vgg19(True)

class NormCalc(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        size = x.size()
        x_view = x.view(size[0], )
        assert False #x_view fix
        avg = x_view.mean(2, keep_dim=True)

        var = x_view - avg
        var = var * var
        var = x_view.mean(2, keep_dim=True)

        return avg, var


class Normalizer(nn.Module):
    def __init__(self):
        self.normcalc = NormCalc()

    def forward(self, x):
        avg, var = self.normcalc(x)
        x = (x - avg) / var
        return x


class AdaIN(nn.Module):
    def __init__(self):
        self.normcalc = NormCalc()
        self.normalizer = Normalizer()

    def forward(self, content, style):
        # todo support multiple styles and mixing

        avg, var = self.normcalc(style)
        content = self.normalizer(content)

        x = (content+avg) * var

        return x

class StyleTransfer:

    class Data:
        epoch = 0
        iter = 0
        ver = 0 # version

    def __init__(self, gpu=False):
        self.use_gpu = torch.cuda.is_available() and gpu

        self.data = StyleTransfer.Data()

        self.max_epoch = 1000000

        self.encoder = self.to_gpu(VGGencoder())
        self.decoder = self.to_gpu(Decoder())

        self.encoder.fix_first_layers("relu4_1")
        self.encoder.get_name_layers()


        self.adain = self.to_gpu(AdaIN())

        self.optimizer = None
        self.optimizer_type = "Adam"

        self.vgg_loss = self.to_gpu(VGGloss())
        self.vgg_loss.reuse(self.encoder)

    def load_state(self, name):
        load_obj = torch.load(name)

        self.data = load_obj["Data"]
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

            if isinstance(x, nn.Module):
                x = nn.DataParallel(x)

        return x

    def to_variable(self, x, volatile=False):
        x = self.to_gpu(x)
        x = Variable(x)

        return x

    def train(self):

        while self.data.epoch < self.max_epoch:
            for idx, (image_content, image_style) in enumerate(self.dataloader):
                image_content = self.to_variable(image_content)
                image_style = self.to_variable(image_style)

                vgg_content = self.encoder(image_content)
                vgg_style = self.encoder(image_style)

                vgg_content_with_stlye = self.adain(vgg_content, vgg_style)

                stylized_content = self.decode(vgg_content_with_stlye)

                style_loss = self.style_loss(stylized_content, image_style)
                content_loss = self.content_loss(stylized_content, vgg_content_with_stlye)

                total_loss = style_loss + content_loss

                print("done")

                self.data.iter += 1

            self.data.epoch +=1


    def create_train_dataset(self):
        crop_size = 256

        trans = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
                ])

        cdataset = csdataset.CSDataset([crop_size,crop_size],"C:/Users/Alex/Desktop/train_1",
                                     "C:/Users/Alex/Desktop/train_1",
                                     trans, trans)


        self.dataloader = torch.utils.data.DataLoader(cdataset, batch_size=4,
                                shuffle=True, num_workers=0)


def main():
    st = StyleTransfer(gpu=True)
    st.create_train_dataset()
    st.train()

if __name__ == "__main__":
    main()