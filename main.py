#!/usr/bin/env python
import torch
import csdataset
from torchvision import transforms

class StyleTransfer:

    class Data:
        epoch = 0
        iter = 0

    def __init__(self, gpu=False):
        self.gpu = gpu

        self.data = StyleTransfer.Data()

        self.max_epoch = 1000000


    def train(self):

        while self.data.epoch < self.max_epoch:
            for idx, data in enumerate(self.dataloader):


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

        cdataset = csdataset.CSDataset([crop_size,crop_size], "/mnt/ssd_data2/alexk/style-transfer/datasets/coco/train2014/",
                                     "/mnt/ssd_data2/alexk/style-transfer/datasets/wikiart/train_1/",
                                     trans, trans)


        self.dataloader = torch.utils.data.DataLoader(cdataset, batch_size=4,
                                shuffle=True, num_workers=0)


def main():
    st = StyleTransfer(gpu=True)
    st.create_train_dataset()
    st.train()

if __name__ == "__main__":
    main()