import torch
import random
import cv2
import os


train_dir = '/data/train_zip'
test_dir = '/data/test_zip'

class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width
        
        #sort images for consistency
        self.imgs = [image for image in sorted(os.listdir(files_dir)) if image[-3:] == 'jpg']
        self.classes = ['_', 'apple', 'banana', 'orange']

    def __getitem__(self, index):
        img_name = self.imgs[index]
        image_path = os.path.join(self.files_dir, img_name)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        
        #divide all pixels rgb vals by 255
        img_res /= 255.0

        #annotation file
        annot_filename = img_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.files_dir, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)

        root = tree.getroot()

        wt = img.shape[1]
        ht = img.shape[0]

        #box coordinates for xml files are extracted
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            #bounding box x coords
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            
            #bounding box y coords
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height

            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        #convert boxes into tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        #areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd

        #image_id
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=image_res, bboxes=target['boxes'], labels=labels)
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img_res, target

    def __len__(self):
        return len(self.imgs)  
    

