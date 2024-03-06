import os, io
import re
import json
import torch
import decord
import torchvision
import numpy as np
import csv


from PIL import Image
from PIL import Image, ImageSequence

from einops import rearrange
from typing import Dict, List, Tuple
from torchvision import transforms
import random




class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        
    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str


class T2VDataset(torch.utils.data.Dataset):
    """Load the UCF101 video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs



        print(f"loading annotations from {configs.csv_path} ...")
        with open(configs.csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.video_folder = configs.video_folder


        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()
        self.video_num = len(self.dataset)

        # ucf101 video frames
        self.video_frame_files = [(os.path.join(self.video_folder, video_dict['video_file_name']),video_dict['prompt']) \
                                  for video_dict in self.dataset]
        random.shuffle(self.video_frame_files)
        self.use_image_num = configs.use_image_num
        self.image_tranform = transforms.Compose([
                transforms.ToTensor(),

                transforms.Resize(configs.image_size, antialias=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.video_frame_num = len(self.video_frame_files)


    def __getitem__(self, index):
        # start_time = time.perf_counter()

        video_index = index % len(self.dataset)



        video_dict = self.dataset[video_index]
        video_file_name = video_dict['video_file_name']
        path = os.path.join(self.video_folder, video_file_name)

        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)
        
        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        # print(start_frame_ind)
        # print(end_frame_ind)
        if total_frames >= self.target_video_len:
            assert end_frame_ind - start_frame_ind >= self.target_video_len, f"{end_frame_ind} - {start_frame_ind} >= { self.target_video_len}"
        frame_indice = np.linspace(start_frame_ind, end_frame_ind, self.target_video_len, dtype=int, endpoint=False)

        assert len(frame_indice) == self.target_video_len
        # print(frame_indice)
        video = vframes[frame_indice] # 这里没有根据步长取视频帧
        # print(type(video))
        # videotransformer data proprecess

        #print(f"before transforem: {video.shape}")
        #print(f"self.transform: {self.transform}")

        video = self.transform(video) # T C H W
        images = []
        image_names = []
        for i in range(self.use_image_num):
            while True:
                try:      
                    video_frame_path, image_name = self.video_frame_files[index+i]

                    frs = [frame.copy().convert('RGB') for frame in ImageSequence.Iterator(Image.open(video_frame_path))]

                    image = frs[random.randint(0, len(frs) - 1)]
                    image = self.image_tranform(image).unsqueeze(0)
                    images.append(image)
                    image_names.append(image_name)
                    break
                except Exception as e:
                    index = random.randint(0, self.video_frame_num - self.use_image_num)
        images =  torch.cat(images, dim=0)
        assert len(images) == self.use_image_num
        assert len(image_names) == self.use_image_num

        image_names = '====='.join(image_names)
        
        #print(f"vieo shape: {video.shape}")
        #print(f"img shape: {images.shape}")
        video_cat = torch.cat([video, images], dim=0)
    
        return {'video': video_cat, 
                'video_name': self.dataset[index]['prompt'],
                'image_name': image_names}

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':

    import argparse
    import video_transforms
    import torch.utils.data as Data
    import torchvision.transforms as transforms
    
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--use-image-num", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--csv_path", type=str, default='')
    parser.add_argument("--video_folder", type=str, default='')


    config = parser.parse_args()


    temporal_sample = video_transforms.TemporalRandomCrop(config.num_frames * config.frame_interval)

    transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.CenterCropResizeVideo(config.image_size),

            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])


    ffs_dataset = GameAnimImages(config, transform=transform_ucf101, temporal_sample=temporal_sample)
    ffs_dataloader = Data.DataLoader(dataset=ffs_dataset, batch_size=1, shuffle=True, num_workers=1)

    # for i, video_data in enumerate(ffs_dataloader):
    sample_count = 20
    for ivd, video_data in enumerate(ffs_dataloader):
        # print(type(video_data))
        video = video_data['video']
        # video_name = video_data['video_name']
        #print(video.shape)
        #print(video_data['image_name'])
        video_name = video_data['video_name']
        image_name = video_data['image_name']
        image_names = []
        for caption in image_name:
            single_caption = [item for item in caption.split('=====')]
            image_names.extend(single_caption)
        #print(image_names)
        #print(video_name)
        #print(video_data[2])

        with open(f'imgname_{ivd}.txt','w') as fp:
            fp.write(';'.join(video_name))
            fp.write(f"\n\n\n")
            fp.write(';'.join(image_names))

        for i in range(20):
            img0 = rearrange(video_data['video'][0][i], 'c h w -> h w c')

            #img0 = rearrange(video_data[0][0][i], 'c h w -> h w c')
            #print('Label: {}'.format(video_data[1]))
            #print(img0.shape)
            img0 = Image.fromarray(np.uint8(img0 * 255))
            img0.save(f'./img_{ivd}_{i}.jpg')

        if ivd > sample_count:
            break
