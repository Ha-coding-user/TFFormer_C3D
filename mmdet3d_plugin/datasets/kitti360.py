import os
import glob
import numpy as np
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmdet.datasets.pipelines import Compose
from mmdet3d.core.bbox.structures import get_box_type

@DATASETS.register_module()
class KITTI360Dataset(Dataset):
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
        box_type_3d='LiDAR'
    ):
        super().__init__()

        self.load_continuous = load_continuous
        self.splits = {
            "train": [
                "2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync",
                "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0007_sync",
                "2013_05_28_drive_0010_sync"
            ],
            "val": ["2013_05_28_drive_0006_sync"],
            "test": ["2013_05_28_drive_0009_sync"]
        }

        self.sequences = self.splits[split]

        self.data_root = data_root
        self.stereo_depth_root = stereo_depth_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.data_infos = self.load_annotations(self.ann_file)
        self.sorted_data_infos = self.sort_data(self.data_infos)
        self.nonempty_data_infos = self.fill_data(self.ann_file)
        self.sorted_nonempty_data_infos = self.sort_data(self.nonempty_data_infos)

        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_map = {'left': '2', 'right': '3'}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]
        
        self.queue_length = queue_length
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.calib = self.read_calib()
        self.cam2world = self.load_cam2world()

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()
    
    def __len__(self):
        return len(self.data_infos)
    
    def check_except_error(self, sorted_data):
        for sequence_id in sorted_data.keys():
            for i in range(len(sorted_data[sequence_id])-1):
                assert int(sorted_data[sequence_id][i]['frame_id']) < int(sorted_data[sequence_id][i+1]['frame_id']), "Error"
                
    def fill_data(self, ann_file=None):
        scans = []
        for sequence in self.sequences:
            calib = self.read_calib()
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            voxel_base_path = os.path.join(self.ann_file, sequence)
            img_base_path = os.path.join(self.data_root, 'data_2d_raw', sequence)

            if self.load_continuous:
                id_base_path = os.path.join(self.data_root, 'data_2d_raw', sequence, 'image_00', 'data_rect', '*.png')
            else:
                id_base_path = os.path.join(self.data_root, 'data_2d_raw', sequence, 'voxels', '*.bin')
            
            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]
                if int(img_id) % 5 == 0:
                    img_2_path = os.path.join(img_base_path, 'image_00', 'data_rect', img_id + '.png')
                    img_3_path = os.path.join(img_base_path, 'image_01', 'data_rect', img_id + '.png')
                    voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')

                    stereo_depth_path = os.path.join(self.stereo_depth_root, "sequences", sequence, img_id + '.npy')

                    if not os.path.exists(voxel_path):
                        voxel_path = None
                
                    scans.append(
                        {   "img_2_path": img_2_path,
                            "img_3_path": img_3_path,
                            "sequence": sequence,
                            "frame_id": img_id,
                            "P2": P2,
                            "P3": P3,
                            "T_velo_2_cam": T_velo_2_cam,
                            "proj_matrix_2": proj_matrix_2,
                            "proj_matrix_3": proj_matrix_3,
                            "voxel_path": voxel_path,
                            # "voxel_1_2_path": voxel_1_2_path,
                            "stereo_depth_path": stereo_depth_path
                        })
        
        return scans
    
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
            cur_input_dict = self.get_data_info(index)
            if cur_input_dict is None:
                return None

            cur_sequence = cur_input_dict['sequence']
            cur_id = cur_input_dict['frame_id']
            history_queue = self.get_history_data_info(cur_sequence, cur_id)
            
            self.pre_pipeline(cur_input_dict)
            example = self.pipeline(cur_input_dict)
            queue = history_queue + [example]
            
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
            cur_input_dict = self.get_data_info(index)
            if cur_input_dict is None:
                return None
            
            cur_sequence = cur_input_dict['sequence']
            cur_id = cur_input_dict['frame_id']
            history_queue = self.get_history_data_info(cur_sequence, cur_id)
            
            self.pre_pipeline(cur_input_dict)
            example = self.pipeline(cur_input_dict)
            queue = history_queue + [example]
            
            return queue
        else:
            input_dict = self.get_data_info(index)
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            
            return [example]
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
        
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
    
    def get_data_info(self, index, sequence_id=None, sorted=False):
        if sorted == True:
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
        sample into includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
            "stereo_depth_path": stereo_depth_path
        '''
        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            sequence = info['sequence'],
            frame_id = info['frame_id'],
        )
        
        pose_list = self.cam2world[info['sequence']]
        cam2world = pose_list[int(info['frame_id'])]

        # load images, intrins, extrins, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []
        cam2world_list = []

        for cam_type in self.camera_used:
            image_paths.append(info['img_{}_path'.format(int(cam_type))])
            lidar2img_rts.append(info['proj_matrix_{}'.format(int(cam_type))])
            cam_intrinsics.append(info['P{}'.format(int(cam_type))])
            lidar2cam_rts.append(info['T_velo_2_cam'])
            cam2world_list.append(cam2world)
        
        focal_length = info['P2'][0, 0]
        baseline = self.dynamic_baseline(info)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                cam2world=cam2world_list,
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
            calib = self.read_calib()
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            voxel_base_path = os.path.join(self.ann_file, sequence)
            img_base_path = os.path.join(self.data_root, 'data_2d_raw', sequence)

            if self.load_continuous:
                id_base_path = os.path.join(self.data_root, 'data_2d_raw', sequence, 'image_00', 'data_rect', '*.png')
            else:
                id_base_path = os.path.join(self.data_root, 'data_2d_raw', sequence, 'voxels', '*.bin')
            
            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]
                img_2_path = os.path.join(img_base_path, 'image_00', 'data_rect', img_id + '.png')
                img_3_path = os.path.join(img_base_path, 'image_01', 'data_rect', img_id + '.png')
                voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')

                stereo_depth_path = os.path.join(self.stereo_depth_root, "sequences", sequence, img_id + '.npy')

                if not os.path.exists(voxel_path):
                    voxel_path = None
                
                scans.append(
                    {   "img_2_path": img_2_path,
                        "img_3_path": img_3_path,
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "P3": P3,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "proj_matrix_3": proj_matrix_3,
                        "voxel_path": voxel_path,
                        # "voxel_1_2_path": voxel_1_2_path,
                        "stereo_depth_path": stereo_depth_path
                    })
        
        return scans

    def get_ann_info(self, index, key='voxel_path'):
        info = self.data_infos[index][key]
        return None if info is None else np.load(info)
    
    @staticmethod
    def read_calib(calib_path=None):
        """
        Tr transforms a point from velodyne coordinates into the 
        left rectified camera coordinate system.
        In order to map a point X from the velodyne scanner to a 
        point x in the i'th image plane, you thus have to transform it like:
        x = Pi * Tr * X
        """
        P2 = np.array([
            [552.554261, 0.000000, 682.049453, 0.000000],
            [0.000000, 552.554261, 238.769549, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000]
        ]).reshape(3, 4)

        P3 = np.array([
            [552.554261, 0.000000, 682.049453, -328.318735],
            [0.000000, 552.554261, 238.769549, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000]
        ]).reshape(3, 4)
        
        R_rect_00 = np.array([
            [0.999974, -0.007141, -0.000089, 0],
            [0.007141, 0.999969, -0.003247, 0],
            [0.000112, 0.003247, 0.999995, 0],
            [0, 0, 0, 1]
        ])

        cam2velo = np.array([   
            [0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
            [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
            [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
            [0, 0, 0, 1]
        ]).reshape(4, 4)
        
        cam2gps = np.array([
            [0.0371783278, -0.0986182135, 0.9944306009, 1.5752681039],
            [0.9992675562, -0.0053553387, -0.0378902567, 0.0043914093],
            [0.0090621821, 0.9951109327, 0.0983468786, -0.6500000000],
            [0, 0, 0, 1]
        ])

        velo2cam = np.linalg.inv(cam2velo)
        calib_out = {}
        calib_out["P2"] = np.identity(4)  # 4x4 matrix
        calib_out["P3"] = np.identity(4)
        calib_out["P2"][:3, :4] = P2.reshape(3, 4)
        calib_out["P3"][:3, :4] = P3.reshape(3, 4)
        calib_out["Tr"] = np.identity(4)
        calib_out["Tr"][:3, :4] = velo2cam[:3, :4]
        calib_out["C2G"] = np.identity(4)
        calib_out["C2G"][:3, :4] = cam2gps[:3, :4]
        calib_out["R_rect"] = R_rect_00
        
        return calib_out
    
    def load_cam2world(self):
        cam2world_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "data_2d_raw", sequence, "poses_match.txt")
            cam2world_dict[sequence] = self.parse_cam2world(pose_path, self.calib)
        
        return cam2world_dict
    
    @staticmethod
    def parse_cam2world(filename, calibration):
        file = open(filename)
        
        cam2world_list = []
        
        Tr = calibration["Tr"]
        C2G = calibration['C2G']
        R_rect = calibration['R_rect']
        
        
        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((3, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            
            # cam2world = np.matmul(np.matmul(pose, C2G), np.linalg.inv(R_rect))
            cam2world = np.matmul(pose, C2G)
            
            cam2world_list.append(cam2world)
            
        return cam2world_list
    
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