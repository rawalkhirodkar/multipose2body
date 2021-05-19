import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torch
import os

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

# -------------------------------------------
class AlignedGlobalDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        input_dir = os.path.join(opt.dataroot, 'input')
        output_dir = os.path.join(opt.dataroot, 'output')

        ### make inputs
        self.person1_dir = os.path.join(input_dir, 'person1')
        self.person2_dir = os.path.join(input_dir, 'person2')
        self.pose_dir = os.path.join(input_dir, 'pose_images')

        ### make outputs
        self.rgb_dir = output_dir

        self.person1_paths = sorted(make_dataset(self.person1_dir))
        self.person2_paths = sorted(make_dataset(self.person2_dir))
        self.pose_paths = sorted(make_dataset(self.pose_dir))
        self.rgb_paths = sorted(make_dataset(self.rgb_dir))

        self.dataset_size = len(self.person1_paths) 

        ## original size
        self.original_size = Image.open(self.person1_paths[0]).size
        self.target_size = (self.original_size[0]/2, self.original_size[1]/2)

        return
      
    def __getitem__(self, index): 
        ## make inputs
        person1_path = self.person1_paths[index]
        person1_image = Image.open(person1_path) ## not it is an rgb image
        person1_image.thumbnail(self.target_size, Image.ANTIALIAS)

        image_name = person1_path.split('/')[-1]

        # person2_path = self.person2_paths[index]
        person2_path = os.path.join(self.person2_dir, image_name)
        person2_image = Image.open(person2_path) ## not it is an rgb image
        person2_image.thumbnail(self.target_size, Image.ANTIALIAS)

        # pose_path = self.pose_paths[index]
        pose_path = os.path.join(self.pose_dir, image_name)
        pose_image = Image.open(pose_path) ## not it is an rgb image
        pose_image.thumbnail(self.target_size, Image.ANTIALIAS)


        ### make transform
        params = get_params(self.opt, person1_image.size)
        transform_input = get_transform(self.opt, params)

        person1_tensor = transform_input(person1_image.convert('RGB'))
        person2_tensor = transform_input(person2_image.convert('RGB'))
        pose_tensor = transform_input(pose_image.convert('RGB'))
        
        output_tensor = inst_tensor = feat_tensor = 0

        ### make outputs
        if self.opt.isTrain:
            # rgb_path = self.rgb_paths[index]
            rgb_path = os.path.join(self.rgb_dir, image_name)   
            rgb = Image.open(rgb_path).convert('RGB')
            rgb.thumbnail(self.target_size, Image.ANTIALIAS)

            transform_output = get_transform(self.opt, params)      
            output_tensor = transform_output(rgb)                          

        input_tensor = torch.cat([pose_tensor, person1_tensor, person2_tensor], dim=0)
        input_dict = {'label': input_tensor, 'inst': inst_tensor, 'image': output_tensor, 
                      'feat': feat_tensor, 'path': pose_path}

        return input_dict

    def __len__(self):
        return len(self.person1_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedGlobalDataset'