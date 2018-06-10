

# https://www.kaggle.com/c/painter-by-numbers/data
# http://images.cocodataset.org/zips/train2014.zip


import torch
import random
import os
from PIL import Image
import torch.utils.data
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CSDataset(torch.utils.data.Dataset):

    class ImageArray:
        def __init__(self, path, size, replace=True, count=0):

            self.count = count
            self.images = self.get_file_list(path)
            # self.images = self.images[:6]
            self.image_valid = set(self.images)
            self.replace = replace
            self.size = size

        def get_file_list(self, path):
            extentsions = ["png", "jpg", "jpeg"]

            ret_list = []
            for cpath, directories, files in os.walk(path):
                for file in files:
                    file_ext =  os.path.splitext(file)[1][1:].lower()

                    if any([ext == file_ext for ext in extentsions]):
                        ret_list.append(os.path.join(cpath, file))

                        if self.count != 0 and len(ret_list) >= self.count:
                            return ret_list


            return sorted(ret_list)

        def __len__(self):
            return len(self.images)

        def random(self):
            return self[random.randint(0, len(self.images)-1)]


        def __getitem__(self, orig_item_id):

            while len(self.image_valid) > 0:

                path = self.images[orig_item_id]

                try:
                    im = Image.open(path)

                    # if im.size[0] < self.size[0] or im.size[1] < self.size[1]:
                    #     print("Bad size", im.size)
                    #     raise Exception("Too small")


                    if im.mode != "RGB":
                        print("mode "+im.mode)
                        im = im.convert("RGB")




                    return im
                except (KeyboardInterrupt, SystemExit) as e:
                    raise e
                except:
                    self.images[orig_item_id] = None

                    print("Bad path {}".format(path))
                    self.image_valid.discard(path)
                    self.images[orig_item_id] = random.sample(self.image_valid, 1)[0]



                if not self.replace:
                    raise Exception("Cannot open file".format())


    @staticmethod
    def get_image_list(path):
        pass

    def __init__(self, size, content_path, style_path, content_transform, style_transform ):
        super(CSDataset, self).__init__()
        self.content_path = content_path
        self.style_path = style_path

        self.content_transform = content_transform
        self.style_transform = style_transform

        self.content_images = CSDataset.ImageArray(self.content_path, size)
        self.style_images = CSDataset.ImageArray(self.style_path, size)

    def __len__(self):
        return len(self.content_images)


    def __getitem__(self, item_id):
        content_img = self.content_images[item_id]
        style_img = self.style_images.random()

        if self.content_transform:
            content_img = self.content_transform(content_img)

        if self.style_transform:
            style_img = self.style_transform(style_img)


        return content_img, style_img






