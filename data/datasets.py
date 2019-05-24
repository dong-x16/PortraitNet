import torch
import torch.utils.data as data
import numpy as np

from datasets_portraitseg import PortraitSeg

class Human(data.Dataset): 
    def __init__(self, exp_args):
        assert exp_args.task in ['seg'], 'Error!, <task> should in [seg]'
        
        self.exp_args = exp_args
        self.task = exp_args.task
        self.datasetlist = exp_args.datasetlist
        
        self.datasets = {}
        self.imagelist = []
        # 
        data_root = '/home/dongx12/Data/'
        root = '/home/dongx12/PortraitNet/data/'
        
        # load dataset
        if 'supervisely' in self.datasetlist:
            ImageRoot = data_root
            AnnoRoot = data_root
            ImgIds_Train = root + 'select_data/supervisely_train_new.txt'
            ImgIds_Test = root + 'select_data/supervisely_test_new.txt'
            exp_args.dataset = 'supervisely'
            self.datasets['supervisely'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)
        
        if 'EG1800' in self.datasetlist:
            ImageRoot = data_root + 'EG1800/Images/'
            AnnoRoot = data_root + 'EG1800/Labels/'
            ImgIds_Train = root + 'select_data/eg1800_train.txt'
            ImgIds_Test = root + 'select_data/eg1800_test.txt'
            exp_args.dataset = 'eg1800'
            self.datasets['eg1800'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)
        
        if 'ATR' in self.datasetlist:
            ImageRoot = data_root + 'ATR/train/images/'
            AnnoRoot = data_root + 'ATR/train/seg/'
            ImgIds_Train = root + 'select_data/ATR_train.txt'
            ImgIds_Test = root + 'select_data/ATR_test.txt'
            exp_args.dataset = 'ATR'
            self.datasets['ATR'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)
        
        if 'supervisely_face_easy' in self.datasetlist:
            ImageRoot = data_root
            AnnoRoot = data_root
            ImgIds_Train = root + 'select_data/supervisely_face_train_easy.txt'
            ImgIds_Test = root + 'select_data/supervisely_face_test_easy.txt'
            exp_args.dataset = 'supervisely_face_easy'
            self.datasets['supervisely_face_easy'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)
            
        if 'supervisely_face_difficult' in self.datasetlist:
            ImageRoot = data_root
            AnnoRoot = data_root
            ImgIds_Train = root + 'select_data/supervisely_face_train_difficult.txt'
            ImgIds_Test = root + 'select_data/supervisely_face_test_difficult.txt'
            exp_args.dataset = 'supervisely_face_difficult'
            self.datasets['supervisely_face_difficult'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)
        
        if 'MscocoBackground' in self.datasetlist:
            dataType = 'train2017'
            ImageRoot = data_root
            AnnoRoot = data_root + 'mscoco2017/annotations/person_keypoints_{}.json'.format(dataType)
            ImgIds_Train = root + 'select_data/select_mscoco_background_train2017.txt'
            ImgIds_Test = root + 'select_data/select_mscoco_background_val2017.txt'
            exp_args.dataset = 'MscocoBackground'
            self.datasets['background'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)

            
        # image list
        for key in self.datasets.keys():
            length = len(self.datasets[key])
            for i in range(length):
                self.imagelist.append([key, i])
        
    def __getitem__(self, index):
        subset, subsetidx = self.imagelist[index]
        
        if self.task == 'seg':
            input_ori, input, output_edge, output_mask = self.datasets[subset][subsetidx]
            return input_ori.astype(np.float32), input.astype(np.float32), \
        output_edge.astype(np.int64), output_mask.astype(np.int64)
           
    def __len__(self):
        return len(self.imagelist)