import os
import glob
import numpy as np
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmdet.datasets.pipelines import Compose
from mmdet3d.core.bbox.structures import get_box_type

import random
import mmcv
from mmcv.parallel import DataContainer as DC
import torch

@DATASETS.register_module()
class SemanticKITTIDataset(Dataset):
    def __init__(
        self,
        data_root,
        stereo_depth_root,
        ann_file,
        pipeline,
        split,
        camera_used,
        occ_size,
        pc_range,
        test_mode=False,
        load_continuous=False,
        queue_length=1,
        bev_size=(200, 200),
        num_classes=20,
        random_camera=False,
        load_multi_voxel=False,
        repeat=1,
        cbgs=False,
        box_type_3d='LiDAR'
    ):
        super().__init__()

        self.load_continuous = load_continuous
        self.splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            # "train": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
            # "train": ["08"],
            "val": ["08"],
            "test": ["08"],
            "test_submit": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }

        self.sequences = self.splits[split]
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_map = {'left': '2', 'right': '3'}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]
        
        # ================== Add: Junwoo ====================================
        self.queue_length       = queue_length     
        self.bev_size           = bev_size         
        self.n_classes          = num_classes       
        self.random_camera      = random_camera
        self.all_camera_ids     = list(self.camera_map.values())
        self.load_multi_voxel   = False
        self.repeat             = repeat
        self.cbgs               = cbgs
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        # ===================================================================  

        self.data_root = data_root
        self.stereo_depth_root = stereo_depth_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.pose_dict = self.get_pose_list()
        self.data_infos = self.load_annotations(self.ann_file)  # calib_path 필요한지 고려
        self.sorted_data_infos = self.sort_data(self.data_infos)
        self.nonempty_data_infos = self.fill_data(self.ann_file)
        self.sorted_nonempty_data_infos = self.sort_data(self.nonempty_data_infos)
        

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def __len__(self):
        return len(self.data_infos)
    
    def check_except_error(self, sorted_data):
        for sequence_id in sorted_data.keys():
            for i in range(len(sorted_data[sequence_id])-1):
                assert int(sorted_data[sequence_id][i]['frame_id']) < int(sorted_data[sequence_id][i+1]['frame_id']), "Error"
    
    def sort_data(self, data_infos):
        sorted_data_infos = {}
        for sequence_id in self.sequences:
            sorted_data_infos[sequence_id] = []
            
        # print('sequence_id:', sorted_data_infos.keys())
            
        # sorted_data = sorted(data_infos, key=lambda x: x['frame_id'])
        for data_info in data_infos:
            sorted_data_infos[data_info['sequence']].append(data_info)
        for sequence_id in sorted_data_infos.keys():
            sorted_data_infos[sequence_id] = sorted(sorted_data_infos[sequence_id], key=lambda x: x['frame_id'])
            
        self.check_except_error(sorted_data_infos)
        # for sequence_id in sorted_data_infos.keys():
        #     print(f"{sequence_id} count: {len(sorted_data_infos[sequence_id])}")
            
        return sorted_data_infos
    
    def fill_data(self, ann_file=None):
        scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "sequences", sequence, "calib.txt")
            )
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            voxel_base_path = os.path.join(self.ann_file, sequence)
            img_base_path = os.path.join(self.data_root, "sequences", sequence)
                        
            id_base_path = os.path.join(self.data_root, "sequences", sequence, 'image_2', '*.png')
            
            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]
                if int(img_id) % 5 == 0:
                    img_2_path = os.path.join(img_base_path, 'image_2', img_id + '.png')
                    img_3_path = os.path.join(img_base_path, 'image_3', img_id + '.png')
                    calib_path = os.path.join(img_base_path, 'calib.txt')
                    voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')
                    stereo_depth_path = os.path.join(self.stereo_depth_root, "sequences", sequence, img_id + '.npy')
                    E = self.pose_dict[sequence][int(img_id)]
                    
                    # for sweep demo or test submission
                    if not os.path.exists(voxel_path):
                        voxel_path = None
                    
                    scans.append(
                        {   "img_2_path": img_2_path,
                            "img_3_path": img_3_path,
                            "sequence": sequence,
                            "frame_id": img_id,
                            "P2": P2,
                            "P3": P3,
                            'E': E,
                            "T_velo_2_cam": T_velo_2_cam,
                            "proj_matrix_2": proj_matrix_2,
                            "proj_matrix_3": proj_matrix_3,
                            "voxel_path": voxel_path,
                            "stereo_depth_path": stereo_depth_path,
                            "calib_path": calib_path,
                        })
                
        return scans  # return to self.data_infos
        
    
    def get_pose_list(self):
        pose_dict = {}
        
        for sequence in self.sequences:
            pose_dict[sequence] = []
            pose_path = os.path.join(self.data_root, 'sequences', sequence, 'poses.txt')
            with open(pose_path, 'r') as f:
                for ext in f.readlines():
                    pose_dict[sequence].append(np.array([float(x) for x in ext.split()]).reshape(3, 4))
                    
        return pose_dict

    def get_history_data_info(self, cur_sequence, cur_id):
        cur_idx = int(cur_id) // 5
        idx_list = [max(cur_idx-3, 0), max(cur_idx-2, 0), max(cur_idx-1, 0)]
        
        history_queue = []
        for idx in idx_list:
            history_dict = self.get_data_info(idx, sequence_id=cur_sequence, sorted=True)
            self.pre_pipeline(history_dict)
            history_example = self.pipeline(history_dict)
            history_queue.append(history_example)
            
        return history_queue
        
       
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        # input_dict = self.get_data_info(index)
        # if input_dict is None:
        #     print('found None in training data')
        #     return None
        
        # example = self.pipeline(input_dict)
        # return example
        if self.queue_length > 1:
            # index_list = list(range(index-self.queue_length, index))
            # random.shuffle(index_list)
            # index_list = sorted(index_list[1:])
            # index_list.append(index)
            cur_input_dict = self.get_data_info(index)
            if cur_input_dict is None:
                return None
            cur_sequence = cur_input_dict['sequence']
            cur_id = cur_input_dict['frame_id']
            history_queue = self.get_history_data_info(cur_sequence, cur_id)
            
            self.pre_pipeline(cur_input_dict)
            example = self.pipeline(cur_input_dict)
            queue = history_queue + [example]
            
            # for  i in index_list:
            #     i = max(0, i)
            #     input_dict = self.get_data_info(i)
            #     if input_dict is None:
            #         return None
            #     self.pre_pipeline(input_dict)
            #     example = self.pipeline(input_dict)
            #     queue.append(example)
            return queue
        else:
            input_dict = self.get_data_info(index)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            return [example]
            
    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        # input_dict = self.get_data_info(index)
        # if input_dict is None:
        #     print('found None in training data')
        #     return None
        
        # example = self.pipeline(input_dict)
        # return example
        if self.queue_length > 1:
            # queue = []
            # index_list = list(range(index-self.queue_length, index))
            # index_list = sorted(index_list[1:])
            # index_list.append(index)
            cur_input_dict = self.get_data_info(index)
            if cur_input_dict is None:
                return None
            cur_sequence = cur_input_dict['sequence']
            cur_id = cur_input_dict['frame_id']
            history_queue = self.get_history_data_info(cur_sequence, cur_id)
            
            self.pre_pipeline(cur_input_dict)
            example = self.pipeline(cur_input_dict)
            queue = history_queue + [example]
            
            # if index < 3:
            #     for i in range(0, len(index_list)):
            #         index_list[i] = index_list[i] if index_list[i] > 0 else 0
                    
            # for i in index_list:
            #     i = max(0, i)
            #     input_dict = self.get_data_info(i)
            #     if input_dict is None:
            #         return None
            #     self.pre_pipeline(input_dict)
            #     example = self.pipeline(input_dict)
            #     queue.append(example)
            return queue
        else:
            input_dict = self.get_data_info(index)
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            return [example]
        
    def union2one(self, queue):
        imgs_list = [each['img_inputs'][0].data for each in queue]
        queue[-1]['img_inputs'] = list(queue[-1]['img_inputs'])
        queue[-1]['img_inputs'][0] = torch.cat(imgs_list, dim=0)
        queue[-1]['img_inputs'] = tuple(queue[-1]['img_inputs'])
        
        imgs_feature = [torch.tensor(np.asarray(each['img_inputs'][-1].data)) for each in queue]
        queue[-1]['img_inputs'] = list(queue[-1]['img_inputs'])
        queue[-1]['img_inputs'][-1] = torch.cat(imgs_feature, dim=0)
        queue[-1]['img_inputs'] = tuple(queue[-1]['img_inputs'])
        
        queue = queue[-1]
        return queue
            
    
    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data
    
    def get_data_info(self, index, sequence_id=None, sorted=False):
        if sorted ==True:
            try:
                info = self.sorted_nonempty_data_infos[sequence_id][index]
            except:
                print(sequence_id)
                print(index)
                print(len(self.sorted_data_infos[sequence_id]))
                exit()
        else:
            info = self.data_infos[index]
        '''
        sample info includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
        '''

        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            sequence = info['sequence'],
            frame_id = info['frame_id'],
        )

        # load images, intrins, extrins, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam2world_rts = []
        cam_intrinsics = []

        for cam_type in self.camera_used:
            image_paths.append(info['img_{}_path'.format(int(cam_type))])
            lidar2img_rts.append(info['proj_matrix_{}'.format(int(cam_type))])
            cam_intrinsics.append(info['P{}'.format(int(cam_type))])
            lidar2cam_rts.append(info['T_velo_2_cam'])
            cam2world_rts.append(info['E'])
        
        focal_length = info['P2'][0, 0]
        baseline = self.dynamic_baseline(info)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                cam2world=cam2world_rts,
                focal_length=focal_length,
                baseline=baseline
            ))
        input_dict['stereo_depth_path'] = info['stereo_depth_path']
        # gt_occ is None for test-set
        input_dict['gt_occ'] = self.get_ann_info(index, key='voxel_path')

        return input_dict
    
    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "sequences", sequence, "calib.txt")
            )
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            voxel_base_path = os.path.join(self.ann_file, sequence)
            img_base_path = os.path.join(self.data_root, "sequences", sequence)
                        
            if self.load_continuous:
                id_base_path = os.path.join(self.data_root, "sequences", sequence, 'image_2', '*.png')
            else:
                id_base_path = os.path.join(self.data_root, "sequences", sequence, 'voxels', '*.bin')
            
            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]
                img_2_path = os.path.join(img_base_path, 'image_2', img_id + '.png')
                img_3_path = os.path.join(img_base_path, 'image_3', img_id + '.png')
                calib_path = os.path.join(img_base_path, 'calib.txt')
                voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')
                stereo_depth_path = os.path.join(self.stereo_depth_root, "sequences", sequence, img_id + '.npy')
                E = self.pose_dict[sequence][int(img_id)]
                
                # for sweep demo or test submission
                if not os.path.exists(voxel_path):
                    voxel_path = None
                
                
                
                scans.append(
                    {   "img_2_path": img_2_path,
                        "img_3_path": img_3_path,
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "P3": P3,
                        'E': E,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "proj_matrix_3": proj_matrix_3,
                        "voxel_path": voxel_path,
                        "stereo_depth_path": stereo_depth_path,
                        "calib_path": calib_path,
                    })
                
        return scans  # return to self.data_infos
    
    def get_ann_info(self, index, key='voxel_path'):
        info = self.data_infos[index][key]
        return np.zeros(self.occ_size) if info is None else np.load(info)
    
    @staticmethod
    def read_calib(calib_path):
        """calib.txt: Calibration data for the cameras: P0/P1 are the 3x4 projection
            matrices after rectification. Here P0 denotes the left and P1 denotes the
            right camera. Tr transforms a point from velodyne coordinates into the
            left rectified camera coordinate system. In order to map a point X from the
            velodyne scanner to a point x in the i'th image plane, you thus have to
            transform it like:
            x = Pi * Tr * X
            - 'image_00': left rectified grayscale image sequence
            - 'image_01': right rectified grayscale image sequence
            - 'image_02': left rectified color image sequence
            - 'image_03': right rectified color image sequence
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out["P2"] = np.identity(4)  # 4x4 matrix
        calib_out["P3"] = np.identity(4)  # 4x4 matrix
        calib_out["P2"][:3, :4] = calib_all["P2"].reshape(3, 4)
        calib_out["P3"][:3, :4] = calib_all["P3"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4) 
        
        return calib_out
    
    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
    
    def dynamic_baseline(self, infos):
        P3 = infos['P3']
        P2 = infos['P2']
        baseline = P3[0,3]/(-P3[0,0]) - P2[0,3]/(-P2[0,0])
        return baseline
    
    def evaluate(self, results, **kwargs):
        results = results['evaluation_semantic']
        for i, class_name in enumerate(results.class_names):
            stats = results.get_stats()
            print("SemIoU/{}".format(class_name), stats["iou_ssc"][i])
        print("mIoU", stats["iou_ssc_mean"])
        print("IoU", stats["iou"])
        print("Precision", stats["precision"])
        print("Recall", stats["recall"])
        results.reset()
        return None