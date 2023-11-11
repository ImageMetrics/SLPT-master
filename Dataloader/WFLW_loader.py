import cv2, copy, logging, os
import numpy as np

import utils

from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class WFLW_test_Dataset(Dataset):
    def __init__(self, cfg, root,
                 transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.root = root
        self.number_landmarks = cfg.WFLW.NUM_POINT
        self.Fraction = cfg.WFLW.FRACTION

        self.flip_index = np.genfromtxt(os.path.join(self.root, "Mirror.txt"),
                                        dtype=int, delimiter=',')

        self.Transform = transform

        # Path to dataset
        self.annotation_file = os.path.join(root, 'WFLW_annotations', 'list_98pt_rect_attr_train_test',
                                            'list_98pt_rect_attr_test.txt')

        self.database = self.get_file_information()

    # Reading Annotation
    def get_file_information(self):
        Data_base = []

        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_point = []
            temp_info = temp_info.split(' ')
            for i in range(2 * self.number_landmarks):
                temp_point.append(float(temp_info[i]))
            point_coord = np.array(temp_point, dtype=np.float).reshape(self.number_landmarks, 2)
            max_index = np.max(point_coord, axis=0)
            min_index = np.min(point_coord, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])
            temp_name = os.path.join(self.root, 'WFLW_images', temp_info[-1])
            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': point_coord})

        return Data_base

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])

        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['point']
        Annotated_Points = Points.copy()

        Img = cv2.imread(Img_path)

        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)

        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

        input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

        for i in range(self.number_landmarks):
            Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)

        meta = {
            'Annotated_Points': Annotated_Points,
            'Img_path': Img_path,
            'Points': Points / (self.Image_size),
            'BBox': BBox,
            'trans': trans,
            'Scale': self.Fraction,
        }

        if self.Transform is not None:
            input = self.Transform(input)

        return input, meta


class WFLW_Dataset(Dataset):
    def __init__(self, cfg, root,
                 transform=None,
                 annotation_file=None,
                 wflw_config=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.root = root

        if wflw_config is None:
            wflw_config = cfg.WFLW

        self.Fraction = wflw_config.FRACTION
        self.number_landmarks = wflw_config.NUM_POINT
        self.flip_index = np.genfromtxt(os.path.join(self.root, "Mirror.txt"),
                                        dtype=int, delimiter=',')

        self.Heatmap_size = cfg.MODEL.HEATMAP

        self.image_dir = wflw_config.IMAGE_DIR

        self.Transform = transform

        if annotation_file is None:
            self.annotation_file = os.path.join(root, 'WFLW_annotations', 'list_98pt_rect_attr_train_test',
                                                    'list_98pt_rect_attr_test.txt')
        else:
            self.annotation_file = annotation_file

        self.database = self.get_file_information()

    def get_file_information(self):
        Data_base = []

        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_point = []
            temp_info = temp_info.split(' ')
            for i in range(2 * self.number_landmarks):
                temp_point.append(float(temp_info[i]))
            point_coord = np.array(temp_point, dtype=np.float).reshape(self.number_landmarks, 2)
            max_index = np.max(point_coord, axis=0)
            min_index = np.min(point_coord, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])
            temp_name = os.path.join(self.root, self.image_dir, temp_info[-1])
            temp_name = temp_name.replace('\\', os.sep)
            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': point_coord})

        return Data_base

    def Image_Flip(self, Img, GT):
        Mirror_GT = []
        width = Img.shape[1]
        for i in self.flip_index:
            Mirror_GT.append([width - 1 - GT[i][0], GT[i][1]])
        Img = cv2.flip(Img, 1)
        return Img, np.array(Mirror_GT)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])
        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['point']
        Annotated_Points = Points.copy()

        Img = cv2.imread(Img_path)

        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

        input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

        for i in range(self.number_landmarks):
            Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)

        meta = {
            "Annotated_Points": Annotated_Points,
            'Img_path': Img_path,
            'Points': Points / (self.Image_size),
            'BBox': BBox,
            'trans': trans,
            'Scale': self.Fraction,
            'angle': 0.0,
            'Translation': [0.0, 0.0],
        }

        # target = np.zeros((self.number_landmarks, self.Heatmap_size, self.Heatmap_size))
        # tpts = Points / (self.Image_size - 1) * (self.Heatmap_size - 1)
        # for i in range(self.number_landmarks):
        #     if tpts[i, 1] > 0:
        #         target[i] = generate_target(target[i], tpts[i], self.sigma)

        if self.Transform is not None:
            input = self.Transform(input)

        return input, meta


class WFLWCal_Dataset(WFLW_Dataset):
    def __init__(self,
                 *args,
                 calibration_annotation_file=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.calibration_annotation_file = calibration_annotation_file

        self.calibration_database = self.get_calibration_file_information()

    def get_calibration_file_information(self):
        Data_base = []

        with open(self.calibration_annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_point = []
            temp_info = temp_info.split(' ')
            for i in range(2 * self.number_landmarks):
                temp_point.append(float(temp_info[i]))
            point_coord = np.array(temp_point, dtype=np.float).reshape(self.number_landmarks, 2)
            max_index = np.max(point_coord, axis=0)
            min_index = np.min(point_coord, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])
            temp_name = os.path.join(self.root, self.image_dir, temp_info[-1])
            temp_name = temp_name.replace('\\', os.sep)
            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': point_coord,})

        return Data_base

    def get_file_information(self):
        Data_base = []

        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_point = []
            temp_info = temp_info.split(' ')
            for i in range(2 * self.number_landmarks):
                temp_point.append(float(temp_info[i]))
            point_coord = np.array(temp_point, dtype=np.float).reshape(self.number_landmarks, 2)
            max_index = np.max(point_coord, axis=0)
            min_index = np.min(point_coord, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])
            temp_name = os.path.join(self.root, self.image_dir, temp_info[-1])
            temp_name = temp_name.replace('\\', os.sep)
            temp_point = []
            for i in range(2 * self.number_landmarks, 4 * self.number_landmarks):
                temp_point.append(float(temp_info[i]))
            start_coord = np.array(temp_point, dtype=np.float).reshape(self.number_landmarks, 2)
            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': point_coord,
                              'start': start_coord,})

        return Data_base

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])
        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['point']
        StartPoints = db_slic['start']
        Annotated_Points = Points.copy()

        Img = cv2.imread(Img_path)

        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

        input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

        for i in range(self.number_landmarks):
            Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)
            StartPoints[i, 0:2] = utils.affine_transform(StartPoints[i, 0:2], trans)

        meta = {
            "Annotated_Points": Annotated_Points,
            'Img_path': Img_path,
            'Points': Points / (self.Image_size),
            'StartPoints': StartPoints / (self.Image_size),
            'BBox': BBox,
            'trans': trans,
            'Scale': self.Fraction,
            'angle': 0.0,
            'Translation': [0.0, 0.0],
        }

        # target = np.zeros((self.number_landmarks, self.Heatmap_size, self.Heatmap_size))
        # tpts = Points / (self.Image_size - 1) * (self.Heatmap_size - 1)
        # for i in range(self.number_landmarks):
        #     if tpts[i, 1] > 0:
        #         target[i] = generate_target(target[i], tpts[i], self.sigma)

        if self.Transform is not None:
            input = self.Transform(input)

        tmp = self.database
        self.database = self.calibration_database
        input_cal, meta_cal = super().__getitem__(idx)
        self.database = tmp

        return input, input_cal, meta, meta_cal

    def Image_Flip(self, Img, GT, ST):
        Mirror_GT = []
        Mirror_ST = []
        width = Img.shape[1]
        for i in self.flip_index:
            Mirror_GT.append([width - 1 - GT[i][0], GT[i][1]])
            Mirror_ST.append([width - 1 - ST[i][0], ST[i][1]])
        Img = cv2.flip(Img, 1)
        return Img, np.array(Mirror_GT), np.array(Mirror_ST)
