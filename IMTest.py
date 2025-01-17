import argparse
import os.path

from Config import cfg
from Config import update_config

from Backbone import get_face_alignment_net

from SLPT import Sparse_alignment_network
from SLPT.Transformer import Transformer

import torch, cv2, math
import numpy as np
from torch import nn

import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import Face_Detector
import utils
import collections

NamedRange = collections.namedtuple('NamedRange', ['name', 'range'])

_CALIBRATION_FRAMES = {
    '1-1 Scale': 110,
    'FaceCapture_Catt_Act6.1Scene1': 179,
    'FaceCapture_Catt_Act7Scene1': 139,
    'FaceCapture_Eddy_Act6.1Scene1': 0,
    'FaceCapture_Eddy_Act10.1Scene1': 0,
    'FaceCapture_Eddy_Act10.5Scene1': 358,
    'Fin_HiNIS_Node_07': 178,
    'Lucas_AFD_Demo_TP_HMC_tk06': 1211,
    'RichardCotton_ROM_Line_Neutral': 27,
    'RichardCotton_TestLine_04': 152,
    'RichardCotton_TestLine_06': 166,
    'RichardCotton_TestLine_09': 82,
    'ROM_CarloMestroni_20221128_055_01_Top': 318,
    'song_BossChick__v1_t5_STa_01_F_STa': 968,
    'song_IcyGRL__v1_t10_STa_01_F_STa': 125,
    'video_2023-06-05_17-26-06': 0,
}
_COMPARISON_REGIONS = {
    '1-1 Scale': [
        NamedRange(name='Brows: non-specific', range=range(192, 245)),
        NamedRange(name='Brows: blink', range=range(335, 364)),
        NamedRange(name='Nose: lighting', range=range(341, 502)),
        NamedRange(name='Brows: lighting', range=range(770, 898)),
        NamedRange(name='Brows: lighting', range=range(934, 976)),
    ],
    'FaceCapture_Catt_Act6.1Scene1': [
        NamedRange(name='Nose: non-specific', range=range(860, 1200)),
        NamedRange(name='Brows: lighting', range=range(4429, 4516)),
        NamedRange(name='Brows: non-specific', range=range(2130, 2325)),
    ],
    'FaceCapture_Catt_Act7Scene1': [
        NamedRange(name='Brows: expression', range=range(435, 890)),
        NamedRange(name='Brows: blink', range=range(0, 200)),
        NamedRange(name='Nose: non-specific', range=range(665, 811)),
        NamedRange(name='Nose: blink', range=range(3505, 3685)),
    ],
    'FaceCapture_Eddy_Act6.1Scene1': [
        NamedRange(name='Brows: expression', range=range(1000, 1310)),
        NamedRange(name='Brows: blink', range=range(194, 229)),
        NamedRange(name='Nose: expression', range=range(1020, 1262)),
        NamedRange(name='Nose: expression', range=range(2774, 3010)),
        NamedRange(name='Brows: non-specific', range=range(3640, 3870)),
    ],
    'FaceCapture_Eddy_Act10.1Scene1': [
        NamedRange(name='Brows: blink', range=range(0, 300)),
        NamedRange(name='Brows: expression', range=range(850, 1100)),
    ],
    'FaceCapture_Eddy_Act10.5Scene1': [
        NamedRange(name='Nose: non-specific', range=range(560, 700)),
        NamedRange(name='Brows: non-specific', range=range(850, 1100)),
    ],
    'Fin_HiNIS_Node_07': [
        NamedRange(name='Brows: expression', range=range(0, 207)),
        NamedRange(name='Brows: expression', range=range(300, 460)),
    ],
    'Lucas_AFD_Demo_TP_HMC_tk06': [
        NamedRange(name='Brows: expression', range=range(0, 309)),
        NamedRange(name='Brows: blink', range=range(349, 381)),
        NamedRange(name='Nose: expression', range=range(223, 345)),
        NamedRange(name='Nose: expression', range=range(450, 610)),
        NamedRange(name='Brows: expression', range=range(940, 1070)),
    ],
    'RichardCotton_ROM_Line_Neutral': [
        NamedRange(name='Brows: expression', range=range(185, 454)),
        NamedRange(name='Brows: blink', range=range(785, 849)),
        NamedRange(name='Nose: expression', range=range(444, 610)),
    ],
    'RichardCotton_TestLine_04': [
        NamedRange(name='Nose: non-specific', range=range(175, 400)),
    ],
    'RichardCotton_TestLine_06': [
        NamedRange(name='Brows: expression', range=range(500, 800)),
        NamedRange(name='Nose: expression', range=range(0, 200)),
    ],
    'RichardCotton_TestLine_09': [
        NamedRange(name='Brows: expression', range=range(400, 535)),
        NamedRange(name='Nose: expression', range=range(150, 250)),
        NamedRange(name='Brows: blink', range=range(580, 620)),
    ],
    'ROM_CarloMestroni_20221128_055_01_Top': [
        NamedRange(name='Brows: expression', range=range(2231, 2365)),
        NamedRange(name='Brows: blink', range=range(18, 205)),
        NamedRange(name='Nose: blink', range=range(320, 430)),
        NamedRange(name='Nose: expression', range=range(2385, 2832)),
    ],
    'song_BossChick__v1_t5_STa_01_F_STa': [
        NamedRange(name='Brows: non-specific', range=range(0, 107)),
        NamedRange(name='Brows: expression', range=range(1093, 1313)),
        NamedRange(name='Nose: non-specific', range=range(2197, 2409)),
    ],
    'song_IcyGRL__v1_t10_STa_01_F_STa': [
        NamedRange(name='Nose: blink', range=range(105, 120)),
        NamedRange(name='Brows: expression', range=range(1210, 1350)),
        NamedRange(name='Brows: expression', range=range(1830, 1870)),
        NamedRange(name='Brows: blink', range=range(0, 200)),
    ],
    'video_2023-06-05_17-26-06': [
        NamedRange(name='Brows: expression', range=range(0, 127)),
        NamedRange(name='Brows: expression', range=range(335, 420)),
        NamedRange(name='Brows: blink', range=range(564, 610)),
    ],
}
ROOT_DIR = r'z:\LocalWorkingRoot\SLPT'


def parse_args():
    parser = argparse.ArgumentParser(description='Video Demo')

    # face detector
    parser.add_argument('-m', '--trained_model', default='./Weight/Face_Detector/yunet_final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--video_source', default='./Video/Video4.mp4', type=str, help='the image file to be detected')
    parser.add_argument('--confidence_threshold', default=0.7, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.3, type=float, help='visualization_threshold')
    parser.add_argument('--base_layers', default=16, type=int, help='the number of the output of the first layer')
    parser.add_argument('--device', default='cuda:0', help='which device the program will run on. cuda:0, cuda:1, ...')

    # landmark detector
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Weight')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='WFLW_6_layer.pth')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    parser.add_argument('--label', help='Label', type=str, default=None)

    args = parser.parse_args()

    return args


def draw_landmark(landmark, image, color=(0, 255, 0)):
    for (x, y) in (landmark + 0.5).astype(np.int32):
        cv2.circle(image, (x, y), 3, color, -1)

    return image


def crop_img(img, bbox, transform):
    x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32)

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    center = np.array([cx, cy])

    scale = max(math.ceil(x2) - math.floor(x1),
                math.ceil(y2) - math.floor(y1)) / 200.0

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    input, trans = utils.crop_v2(img, center, scale * 1.15, (256, 256))

    input = transform(input).unsqueeze(0)

    return input, trans


def face_detection(img, model, im_width, im_height):
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
    img = np.float32(img)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    scale = torch.Tensor([im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height])
    scale = scale.to(device)

    # feed forward
    loc, conf, iou = model(img)

    # post processing
    priorbox = Face_Detector.PriorBox(Face_Detector.cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = Face_Detector.decode(loc.data.squeeze(0), prior_data, Face_Detector.cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    cls_scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    iou_scores = iou.squeeze(0).data.cpu().numpy()[:, 0]
    # clamp here for the compatibility for ONNX
    _idx = np.where(iou_scores < 0.)
    iou_scores[_idx] = 0.
    _idx = np.where(iou_scores > 1.)
    iou_scores[_idx] = 1.
    scores = np.sqrt(cls_scores * iou_scores)

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    selected_idx = np.array([0, 1, 2, 3, 14])
    keep = Face_Detector.nms(dets[:, selected_idx], args.nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]

    return dets


def find_max_box(box_array):
    potential_box = []
    for b in box_array:
        if b[14] < args.vis_thres:
            continue
        potential_box.append(np.array([b[0], b[1], b[2], b[3], b[14]], dtype=np.int))

    if len(potential_box) > 0:
        x1, y1, x2, y2 = (potential_box[0][:4]).astype(np.int32)
        Max_box = (x2 - x1) * (y2 - y1)
        Max_index = 0
        for index in range(1, len(potential_box)):
            x1, y1, x2, y2 = (potential_box[index][:4]).astype(np.int32)
            temp_box = (x2 - x1) * (y2 - y1)
            if temp_box >= Max_box:
                Max_box = temp_box
                Max_index = index
        return box_array[Max_index]
    else:
        return None


class Sparse_alignment_network_refine(Sparse_alignment_network):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, image, landmarks_1=None, landmarks_2=None):
        bs = image.size(0)

        output_list = []

        feature_map = self.backbone(image)

        if landmarks_1 is None:
            initial_landmarks = self.initial_points.repeat(bs, 1, 1).to(image.device)

            # stage_1
            ROI_anchor_1, bbox_size_1, start_anchor_1 = self.ROI_1(initial_landmarks.detach())
            ROI_anchor_1 = ROI_anchor_1.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
            ROI_feature_1 = self.interpolation(feature_map, ROI_anchor_1.detach()).view(bs, self.num_point,
                                                                                        self.Sample_num,
                                                                                        self.Sample_num, self.d_model)
            ROI_feature_1 = ROI_feature_1.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                               self.d_model).permute(0, 3, 2, 1)

            transformer_feature_1 = self.feature_extractor(ROI_feature_1).view(bs, self.num_point, self.d_model)

            offset_1 = self.Transformer(transformer_feature_1)
            offset_1 = self.out_layer(offset_1)

            landmarks_1 = start_anchor_1.unsqueeze(1) + bbox_size_1.unsqueeze(1) * offset_1
            output_list.append(landmarks_1)

        if landmarks_2 is None:
            # stage_2
            ROI_anchor_2, bbox_size_2, start_anchor_2 = self.ROI_2(landmarks_1[:, -1, :, :].detach())
            ROI_anchor_2 = ROI_anchor_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
            ROI_feature_2 = self.interpolation(feature_map, ROI_anchor_2.detach()).view(bs, self.num_point,
                                                                                        self.Sample_num,
                                                                                        self.Sample_num, self.d_model)
            ROI_feature_2 = ROI_feature_2.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                               self.d_model).permute(0, 3, 2, 1)

            transformer_feature_2 = self.feature_extractor(ROI_feature_2).view(bs, self.num_point, self.d_model)

            offset_2 = self.Transformer(transformer_feature_2)
            offset_2 = self.out_layer(offset_2)

            landmarks_2 = start_anchor_2.unsqueeze(1) + bbox_size_2.unsqueeze(1) * offset_2
            output_list.append(landmarks_2)

        # stage_3
        ROI_anchor_3, bbox_size_3, start_anchor_3 = self.ROI_3(landmarks_2[:, -1, :, :].detach())
        ROI_anchor_3 = ROI_anchor_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_3 = self.interpolation(feature_map, ROI_anchor_3.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                    self.Sample_num, self.d_model)
        ROI_feature_3 = ROI_feature_3.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                           self.d_model).permute(0, 3, 2, 1)

        transformer_feature_3 = self.feature_extractor(ROI_feature_3).view(bs, self.num_point, self.d_model)

        offset_3 = self.Transformer(transformer_feature_3)
        offset_3 = self.out_layer(offset_3)

        landmarks_3 = start_anchor_3.unsqueeze(1) + bbox_size_3.unsqueeze(1) * offset_3
        output_list.append(landmarks_3)

        return output_list


class TransformerCal(Transformer):
    def __init__(self, num_points, d_model=256, nhead=8, num_decoder_layer=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", normalize_before=True):
        super().__init__(num_points, d_model=d_model, nhead=nhead,
                         num_decoder_layer=num_decoder_layer, dim_feedforward=dim_feedforward,
                         dropout=dropout, activation=activation, normalize_before=normalize_before)
        # calibration encoding
        # self.calibration_encoding = nn.Parameter(torch.randn(1, 2, self.d_model))
        # self.frame_encoding = nn.Parameter(torch.randn(1, 2, self.d_model))

        self.calibration_encoding = nn.Parameter(torch.randn(1, num_points, d_model))

        self._reset_parameters()

    def forward(self, src, cal):
        bs, num_feat, len_feat = src.size()

        structure_encoding = self.structure_encoding.repeat(bs, 1, 1).permute(1, 0, 2)
        calibration_encoding = self.calibration_encoding.repeat(bs, 1, 1).permute(1, 0, 2)

        # calibration_encoding = (
        #         structure_encoding * self.calibration_encoding[:, 0, :].repeat(bs, num_feat, 1).permute(1, 0, 2)
        # ) + self.calibration_encoding[:, 1, :].repeat(bs, num_feat, 1).permute(1, 0, 2)
        # structure_encoding = (
        #         structure_encoding * self.frame_encoding[:, 0, :].repeat(bs, num_feat, 1).permute(1, 0, 2)
        # ) + self.frame_encoding[:, 1, :].repeat(bs, num_feat, 1).permute(1, 0, 2)

        landmark_query = self.landmark_query.repeat(bs, 1, 1).permute(1, 0, 2)

        src = src.permute(1, 0, 2)
        cal = cal.permute(1, 0, 2)

        src_cal = torch.cat((src, cal), dim=0)
        src_cal_encoding = torch.cat((structure_encoding, calibration_encoding), dim=0)

        tgt = torch.zeros_like(landmark_query)
        tgt = self.Transformer_block(tgt, src_cal,
                                     pos=src_cal_encoding,
                                     query_pos=landmark_query)

        return tgt.permute(2, 0, 1, 3)


class Sparse_alignment_network_cal(Sparse_alignment_network):
    def __init__(self, num_point, d_model, trainable,
                 return_interm_layers, dilation, nhead, feedforward_dim,
                 initial_path, cfg):
        super().__init__(num_point, d_model, trainable,
                         return_interm_layers, dilation, nhead, feedforward_dim,
                         initial_path, cfg)

        # Transformer
        self.Transformer = TransformerCal(num_point, d_model, nhead, cfg.TRANSFORMER.NUM_DECODER,
                                          feedforward_dim, dropout=0.1)

        self.feature_extractor_cal = nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=False)

        self._reset_parameters()

        # backbone
        self.backbone = get_face_alignment_net(cfg)

    def calibrate(self, cal_image, cal_landmarks):

        bs = cal_image.size(0)
        calibration_feature_map = self.backbone(cal_image)

        # cal features
        ROI_feature_cal_1, _, _ = self.get_image_features(calibration_feature_map, cal_landmarks, stage=1)
        ROI_feature_cal_2, _, _ = self.get_image_features(calibration_feature_map, cal_landmarks, stage=2)
        ROI_feature_cal_3, _, _ = self.get_image_features(calibration_feature_map, cal_landmarks, stage=3)

        transformer_feature_cal_1 = self.feature_extractor_cal(ROI_feature_cal_1).view(bs, self.num_point, self.d_model)
        transformer_feature_cal_2 = self.feature_extractor_cal(ROI_feature_cal_2).view(bs, self.num_point, self.d_model)
        transformer_feature_cal_3 = self.feature_extractor_cal(ROI_feature_cal_3).view(bs, self.num_point, self.d_model)

        self.transformer_feature_cal_1 = transformer_feature_cal_1
        self.transformer_feature_cal_2 = transformer_feature_cal_2
        self.transformer_feature_cal_3 = transformer_feature_cal_3

    def forward(self, image):

        bs = image.size(0)

        output_list = []

        feature_map = self.backbone(image)

        initial_landmarks = self.initial_points.repeat(bs, 1, 1).to(image.device)

        # stage_1
        ROI_feature_1, bbox_size_1, start_anchor_1 = \
            self.get_image_features(feature_map, initial_landmarks, stage=1)

        transformer_feature_1 = self.feature_extractor(ROI_feature_1).view(bs, self.num_point, self.d_model)

        offset_1 = self.Transformer(transformer_feature_1, self.transformer_feature_cal_1)
        offset_1 = self.out_layer(offset_1)

        landmarks_1 = start_anchor_1.unsqueeze(1) + bbox_size_1.unsqueeze(1) * offset_1
        output_list.append(landmarks_1)

        # stage_2
        ROI_feature_2, bbox_size_2, start_anchor_2 = \
            self.get_image_features(feature_map, landmarks_1[:, -1, :, :], stage=2)

        transformer_feature_2 = self.feature_extractor(ROI_feature_2).view(bs, self.num_point, self.d_model)

        offset_2 = self.Transformer(transformer_feature_2, self.transformer_feature_cal_2)
        offset_2 = self.out_layer(offset_2)

        landmarks_2 = start_anchor_2.unsqueeze(1) + bbox_size_2.unsqueeze(1) * offset_2
        output_list.append(landmarks_2)

        # stage_3
        ROI_feature_3, bbox_size_3, start_anchor_3 = \
            self.get_image_features(feature_map, landmarks_2[:, -1, :, :], stage=3)

        transformer_feature_3 = self.feature_extractor(ROI_feature_3).view(bs, self.num_point, self.d_model)

        offset_3 = self.Transformer(transformer_feature_3, self.transformer_feature_cal_3)
        offset_3 = self.out_layer(offset_3)

        landmarks_3 = start_anchor_3.unsqueeze(1) + bbox_size_3.unsqueeze(1) * offset_3
        output_list.append(landmarks_3)

        return output_list

    def get_image_features(self, feature_map, landmarks, stage=1):
        bs = feature_map.size(0)

        # features
        if stage == 1:
            ROI_anchor_1, bbox_size_1, start_anchor_1 = self.ROI_1(landmarks.detach())
            ROI_anchor_1 = ROI_anchor_1.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
            ROI_feature_1 = self.interpolation(feature_map, ROI_anchor_1.detach()).view(bs, self.num_point,
                                                                                        self.Sample_num,
                                                                                        self.Sample_num, self.d_model)
            ROI_feature_1 = ROI_feature_1.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                               self.d_model).permute(0, 3, 2, 1)
            return ROI_feature_1, bbox_size_1, start_anchor_1

        elif stage == 2:
            ROI_anchor_2, bbox_size_2, start_anchor_2 = self.ROI_2(landmarks.detach())
            ROI_anchor_2 = ROI_anchor_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
            ROI_feature_2 = self.interpolation(feature_map, ROI_anchor_2.detach()).view(bs, self.num_point,
                                                                                        self.Sample_num,
                                                                                        self.Sample_num, self.d_model)
            ROI_feature_2 = ROI_feature_2.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                               self.d_model).permute(0, 3, 2, 1)
            return ROI_feature_2, bbox_size_2, start_anchor_2
        else:
            ROI_anchor_3, bbox_size_3, start_anchor_3 = self.ROI_3(landmarks.detach())
            ROI_anchor_3 = ROI_anchor_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
            ROI_feature_3 = self.interpolation(feature_map, ROI_anchor_3.detach()).view(bs, self.num_point,
                                                                                        self.Sample_num,
                                                                                        self.Sample_num, self.d_model)
            ROI_feature_3 = ROI_feature_3.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                               self.d_model).permute(0, 3, 2, 1)

            return ROI_feature_3, bbox_size_3, start_anchor_3


class Sparse_alignment_network_cal_refine(Sparse_alignment_network_cal):
    def __init__(self, num_point, d_model, trainable,
                 return_interm_layers, dilation, nhead, feedforward_dim,
                 initial_path, cfg):
        super().__init__(num_point, d_model, trainable,
                         return_interm_layers, dilation, nhead, feedforward_dim,
                         initial_path, cfg)

    def forward(self, image, initial_landmarks,
                landmarks_1=None, landmarks_2=None):
        bs = image.size(0)

        output_list = []

        feature_map = self.backbone(image)

        # stage_1
        ROI_feature_1, bbox_size_1, start_anchor_1 = \
            self.get_image_features(feature_map, initial_landmarks, stage=1)

        transformer_feature_1 = self.feature_extractor(ROI_feature_1).view(bs, self.num_point, self.d_model)

        offset_1 = self.Transformer(transformer_feature_1, self.transformer_feature_cal_1)
        offset_1 = self.out_layer(offset_1)

        landmarks_1 = start_anchor_1.unsqueeze(1) + bbox_size_1.unsqueeze(1) * offset_1
        output_list.append(landmarks_1)

        # stage_2
        ROI_feature_2, bbox_size_2, start_anchor_2 = \
            self.get_image_features(feature_map, landmarks_1[:, -1, :, :], stage=2)

        transformer_feature_2 = self.feature_extractor(ROI_feature_2).view(bs, self.num_point, self.d_model)

        offset_2 = self.Transformer(transformer_feature_2, self.transformer_feature_cal_2)
        offset_2 = self.out_layer(offset_2)

        landmarks_2 = start_anchor_2.unsqueeze(1) + bbox_size_2.unsqueeze(1) * offset_2
        output_list.append(landmarks_2)

        # stage_3
        ROI_feature_3, bbox_size_3, start_anchor_3 = \
            self.get_image_features(feature_map, landmarks_2[:, -1, :, :], stage=3)

        transformer_feature_3 = self.feature_extractor(ROI_feature_3).view(bs, self.num_point, self.d_model)

        offset_3 = self.Transformer(transformer_feature_3, self.transformer_feature_cal_3)
        offset_3 = self.out_layer(offset_3)

        landmarks_3 = start_anchor_3.unsqueeze(1) + bbox_size_3.unsqueeze(1) * offset_3
        output_list.append(landmarks_3)

        return output_list


def run_with_detector(image_files, cfg, net, normalize, model):
    for image_file in image_files:
        frame = cv2.imread(image_file)
        im_width = frame.shape[1]
        im_height = frame.shape[0]

        dets = face_detection(frame.copy(), net, 320, 240)
        bbox = find_max_box(dets)

        if bbox is not None:
            bbox[0] = int(bbox[0] / 320.0 * im_width + 0.5)
            bbox[2] = int(bbox[2] / 320.0 * im_width + 0.5)
            bbox[1] = int(bbox[1] / 240.0 * im_height + 0.5)
            bbox[3] = int(bbox[3] / 240.0 * im_height + 0.5)
            alignment_input, trans = crop_img(frame.copy(), bbox, normalize)

            outputs_initial = model(alignment_input.cuda())
            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            landmark = utils.transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)

            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            frame = draw_landmark(landmark, frame)
            # out.write(frame)
            cv2.imshow('res', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def run_with_detector_cal(image_files, cal_image_file, cal_image_landmarks,
                          cfg, net, normalize, model):
    frame = cv2.imread(cal_image_file)

    bbox = get_bbox(cal_image_landmarks, cfg)
    cal_input, trans = crop_img(frame.copy(), bbox, normalize)

    cal_landmarks_model = torch.from_numpy(
        utils.transform_pixel_v2(cal_image_landmarks, trans) / cfg.MODEL.IMG_SIZE
    ).view(1, cal_image_landmarks.shape[0], cal_image_landmarks.shape[1]).float()

    model.calibrate(cal_input.cuda(), cal_landmarks_model.cuda())

    for image_file in image_files:
        frame = cv2.imread(image_file)
        im_width = frame.shape[1]
        im_height = frame.shape[0]

        dets = face_detection(frame.copy(), net, 320, 240)
        bbox = find_max_box(dets)

        if bbox is not None:
            bbox[0] = int(bbox[0] / 320.0 * im_width + 0.5)
            bbox[2] = int(bbox[2] / 320.0 * im_width + 0.5)
            bbox[1] = int(bbox[1] / 240.0 * im_height + 0.5)
            bbox[3] = int(bbox[3] / 240.0 * im_height + 0.5)
            alignment_input, trans = crop_img(frame.copy(), bbox, normalize)

            outputs_initial = model(alignment_input.cuda())
            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            landmark = utils.transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)

            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            frame = draw_landmark(landmark, frame)
            # out.write(frame)
            cv2.imshow('res', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def get_bbox(landmarks, cfg):
    max_index = np.max(landmarks, axis=0)
    min_index = np.min(landmarks, axis=0)
    bbox = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                     max_index[1] - min_index[1]])

    # scale up by average factor
    scale_fac = cfg.HEADCAM.FRACTION / 1.15  # a 1.15 scale is applied later
    center_point = bbox[0:2] + (bbox[2:4] / 2)
    wh_scaled = bbox[2:4] * scale_fac
    wh_scaled[:] = np.max(wh_scaled)
    wh_scaled += 1  # make sure the box isn't 0 width

    # output bound box is [pt1x, pt1y, pt2x, pt2y]
    bbox_scaled = bbox
    bbox_scaled[0:2] = center_point - (wh_scaled * 0.5)
    bbox_scaled[2:4] = center_point + (wh_scaled * 0.5)

    return bbox_scaled


def run_bbox_cal(image_files, image_landmarks, cal_image_file, cal_image_landmarks,
                          cfg, normalize, model, label):

    display = True

    output_dir = os.path.join(ROOT_DIR, 'Results', label)
    os.makedirs(output_dir, exist_ok=True)
    refined_landmarks = image_landmarks
    output_file = os.path.basename(os.path.dirname(image_files[0]))
    print(f'Processing {output_file}')
    output_file = os.path.join(output_dir, f'{output_file}.npz')

    frame = cv2.imread(cal_image_file)

    bbox = get_bbox(cal_image_landmarks, cfg)
    cal_input, trans = crop_img(frame.copy(), bbox, normalize)

    cal_landmarks_model = torch.from_numpy(
        utils.transform_pixel_v2(cal_image_landmarks, trans) / cfg.MODEL.IMG_SIZE
    ).view(1, cal_image_landmarks.shape[0], cal_image_landmarks.shape[1]).float()

    model.calibrate(cal_input.cuda(), cal_landmarks_model.cuda())

    import tqdm
    for i, (image_file, landmarks) in tqdm.tqdm(enumerate(zip(image_files, image_landmarks))):
        frame = cv2.imread(image_file)

        bbox = get_bbox(landmarks, cfg)
        alignment_input, trans = crop_img(frame.copy(), bbox, normalize)

        outputs_initial = model(alignment_input.cuda())
        output = outputs_initial[2][0, -1, :, :].cpu().numpy()

        landmark = utils.transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)
        refined_landmarks[i, :, :] = landmark

        if display:
            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            frame = draw_landmark(landmark, frame)
            # out.write(frame)
            cv2.imshow('res', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    np.savez(output_file, image_files=image_files, landmarks=refined_landmarks)


def run_refinement(image_files, image_landmarks, cfg, normalize, model, label):
    redo_track = True
    display = False

    output_dir = os.path.join(ROOT_DIR, 'Results', label)
    os.makedirs(output_dir, exist_ok=True)
    refined_landmarks = image_landmarks
    output_file = os.path.basename(os.path.dirname(image_files[0]))
    print(f'Processing {output_file}')
    output_file = os.path.join(output_dir, f'{output_file}.npz')

    if not redo_track and os.path.exists(output_file):
        return

    import tqdm
    for i, (image_file, landmarks) in tqdm.tqdm(enumerate(zip(image_files, image_landmarks))):
        frame = cv2.imread(image_file)

        bbox = get_bbox(landmarks, cfg)
        alignment_input, trans = crop_img(frame.copy(), bbox, normalize)

        landmarks_model = torch.from_numpy(
            utils.transform_pixel_v2(landmarks, trans) / cfg.MODEL.IMG_SIZE
        ).view(1, 1, landmarks.shape[0], landmarks.shape[1]).float()

        outputs_initial = model(alignment_input.cuda(),
                                landmarks_1=landmarks_model.cuda(),
                                landmarks_2=None,
                                )
        output = outputs_initial[-1][0, -1, :, :].cpu().numpy()

        landmark = utils.transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)

        refined_landmarks[i, :, :] = landmark

        if display:
            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            frame = draw_landmark(landmarks, frame, (255, 0, 0))
            frame = draw_landmark(landmark, frame)
            # out.write(frame)
            cv2.imshow('res', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    np.savez(output_file, image_files=image_files, landmarks=refined_landmarks)


def run_refinement_cal(image_files, image_landmarks,
                       cal_image_file, cal_image_landmarks, cfg, normalize, model, label):
    redo_track = True
    display = False

    output_dir = os.path.join(ROOT_DIR, 'Results', label)
    os.makedirs(output_dir, exist_ok=True)
    refined_landmarks = image_landmarks
    output_file = os.path.basename(os.path.dirname(image_files[0]))
    print(f'Processing {output_file}')
    output_file = os.path.join(output_dir, f'{output_file}.npz')

    if not redo_track and os.path.exists(output_file):
        return

    frame = cv2.imread(cal_image_file)

    bbox = get_bbox(cal_image_landmarks, cfg)
    cal_input, trans = crop_img(frame.copy(), bbox, normalize)

    cal_landmarks_model = torch.from_numpy(
        utils.transform_pixel_v2(cal_image_landmarks, trans) / cfg.MODEL.IMG_SIZE
    ).view(1, cal_image_landmarks.shape[0], cal_image_landmarks.shape[1]).float()

    model.calibrate(cal_input.cuda(), cal_landmarks_model.cuda())

    import tqdm
    for i, (image_file, landmarks) in tqdm.tqdm(enumerate(zip(image_files, image_landmarks)), total=len(image_files)):
        frame = cv2.imread(image_file)

        bbox = get_bbox(landmarks, cfg)
        alignment_input, trans = crop_img(frame.copy(), bbox, normalize)

        landmarks_model = torch.from_numpy(
            utils.transform_pixel_v2(landmarks, trans) / cfg.MODEL.IMG_SIZE
        ).view(1, landmarks.shape[0], landmarks.shape[1]).float()

        outputs_initial = model(alignment_input.cuda(),
                                landmarks_model.cuda(),
                                landmarks_1=None,
                                landmarks_2=None,
                                )
        output = outputs_initial[-1][0, -1, :, :].cpu().numpy()

        landmark = utils.transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)

        refined_landmarks[i, :, :] = landmark

        if display:
            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            frame = draw_landmark(landmarks, frame, (255, 0, 0))
            frame = draw_landmark(landmark, frame)
            # out.write(frame)
            cv2.imshow('res', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    np.savez(output_file, image_files=image_files, landmarks=refined_landmarks)


def run_frame_to_frame(image_files, image_landmarks, cal_image_file, cal_image_landmarks,
                       cfg, normalize, model, label):
    redo_track = True
    display = True

    output_dir = os.path.join(ROOT_DIR, 'Results', label)
    os.makedirs(output_dir, exist_ok=True)
    refined_landmarks = image_landmarks
    output_file = os.path.basename(os.path.dirname(image_files[0]))
    print(f'Processing {output_file}')
    output_file = os.path.join(output_dir, f'{output_file}.npz')

    if not redo_track and os.path.exists(output_file):
        return

    frame = cv2.imread(cal_image_file)

    bbox = get_bbox(cal_image_landmarks, cfg)
    cal_input, trans = crop_img(frame.copy(), bbox, normalize)

    cal_landmarks_model = torch.from_numpy(
        utils.transform_pixel_v2(cal_image_landmarks, trans) / cfg.MODEL.IMG_SIZE
    ).view(1, cal_image_landmarks.shape[0], cal_image_landmarks.shape[1]).float()

    model.calibrate(cal_input.cuda(), cal_landmarks_model.cuda())

    import tqdm
    prev_landmarks = image_landmarks[0]
    for i, image_file in tqdm.tqdm(enumerate(image_files)):
        frame = cv2.imread(image_file)

        bbox = get_bbox(prev_landmarks, cfg)
        alignment_input, trans = crop_img(frame.copy(), bbox, normalize)

        landmarks_model = torch.from_numpy(
            utils.transform_pixel_v2(prev_landmarks, trans) / cfg.MODEL.IMG_SIZE
        ).view(1, prev_landmarks.shape[0], prev_landmarks.shape[1]).float()

        outputs_initial = model(alignment_input.cuda(),landmarks_model.cuda())
        output = outputs_initial[-1][0, -1, :, :].cpu().numpy()

        landmark = utils.transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)

        refined_landmarks[i, :, :] = landmark
        prev_landmarks = landmark

        if display:
            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            frame = draw_landmark(landmark, frame)
            # out.write(frame)
            cv2.imshow('res', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    np.savez(output_file, image_files=image_files, landmarks=refined_landmarks)


device = None
args = None


def main():
    method = 'refinement_cal'  # bbox_cal, refinement, refinement_cal, detector, detector_cal, frame_to_frame
    track_only_regions = True

    global args
    args = parse_args()
    update_config(cfg, args)

    global device
    device = torch.device(args.device)

    torch.set_grad_enabled(False)

    # Cuda
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # load face detector
    net = Face_Detector.YuFaceDetectNet(phase='test', size=None)  # initialize detector
    net = Face_Detector.load_model(net, args.trained_model, True)
    net.eval()
    net = net.to(device)
    print('Finished loading Face Detector!')

    if method in ['refinement', ]:
        model = Sparse_alignment_network_refine(cfg.HEADCAM.NUM_POINT, cfg.MODEL.OUT_DIM,
                                                cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                                cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                                cfg.TRANSFORMER.FEED_DIM, cfg.HEADCAM.INITIAL_PATH, cfg)
    elif method in ['refinement_cal', 'frame_to_frame', ]:
        model = Sparse_alignment_network_cal_refine(cfg.HEADCAMCAL.NUM_POINT, cfg.MODEL.OUT_DIM,
                                                    cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                                    cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                                    cfg.TRANSFORMER.FEED_DIM, cfg.HEADCAMCAL.INITIAL_PATH, cfg)
    elif method in ['detector_cal', 'bbox_cal']:
        model = Sparse_alignment_network_cal(cfg.HEADCAMCAL.NUM_POINT, cfg.MODEL.OUT_DIM,
                                             cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                             cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                             cfg.TRANSFORMER.FEED_DIM, cfg.HEADCAMCAL.INITIAL_PATH, cfg)
    else:
        model = Sparse_alignment_network(cfg.HEADCAM.NUM_POINT, cfg.MODEL.OUT_DIM,
                                         cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                         cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                         cfg.TRANSFORMER.FEED_DIM, cfg.HEADCAM.INITIAL_PATH, cfg)
    model.cuda()

    checkpoint_file = os.path.join(args.modelDir, args.checkpoint)
    checkpoint = torch.load(checkpoint_file)
    pretrained_dict = {k: v for k, v in checkpoint.items()
                       if k in model.state_dict().keys()}
    model.load_state_dict(pretrained_dict)
    model.eval()

    print('Finished loading face landmark detector')

    # test data file
    import glob
    test_files = glob.glob(os.path.join(ROOT_DIR, r'Results\LDSDK\*.npz'))

    for test_data_file in test_files:
        npz_file = np.load(test_data_file)
        if track_only_regions:
            video_name = os.path.basename(test_data_file)[:-11]
            if video_name not in _COMPARISON_REGIONS.keys():
                raise RuntimeError(f'Missing video regions: {video_name}')
            image_files = []
            landmarks = []
            for region in _COMPARISON_REGIONS[video_name]:
                image_files.extend(
                    npz_file['image_files'][region.range]
                )
                landmarks.append(
                    npz_file['landmarks'][region.range, :, :]
                )
            landmarks = np.concatenate(landmarks, axis=0)
        else:
            image_files = npz_file['image_files']
            landmarks = npz_file['landmarks']

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        normalize = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        if method == 'refinement':
            run_refinement(image_files, landmarks, cfg, normalize, model, args.label)
        elif method == 'frame_to_frame':
            video_name = os.path.basename(test_data_file)[:-11]
            cal_ind = _CALIBRATION_FRAMES[video_name]
            cal_image_file = npz_file['image_files'][cal_ind]
            cal_image_landmarks = npz_file['landmarks'][cal_ind, :, :]
            run_frame_to_frame(image_files, landmarks, cal_image_file, cal_image_landmarks, cfg, normalize, model, args.label)
        elif method == 'refinement_cal':
            video_name = os.path.basename(test_data_file)[:-11]
            cal_ind = _CALIBRATION_FRAMES[video_name]
            cal_image_file = npz_file['image_files'][cal_ind]
            cal_image_landmarks = npz_file['landmarks'][cal_ind, :, :]
            run_refinement_cal(image_files, landmarks, cal_image_file, cal_image_landmarks, cfg, normalize, model,
                               args.label)
        elif method == 'detector':
            run_with_detector(image_files, cfg, net, normalize, model)
        elif method == 'detector_cal':
            video_name = os.path.basename(test_data_file)[:-11]
            cal_ind = _CALIBRATION_FRAMES[video_name]
            cal_image_file = npz_file['image_files'][cal_ind]
            cal_image_landmarks = npz_file['landmarks'][cal_ind, :, :]
            run_with_detector_cal(image_files, cal_image_file, cal_image_landmarks, cfg, net, normalize, model)
        elif method == 'bbox_cal':
            video_name = os.path.basename(test_data_file)[:-11]
            cal_ind = _CALIBRATION_FRAMES[video_name]
            cal_image_file = npz_file['image_files'][cal_ind]
            cal_image_landmarks = npz_file['landmarks'][cal_ind, :, :]
            run_bbox_cal(image_files, landmarks, cal_image_file, cal_image_landmarks, cfg, normalize, model, args.label)
        else:
            raise RuntimeError('Unknown method')


def calcuate_loss(name, pred, gt, trans):
    pred = (pred - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)

    if name == 'HEADCAMCAL':
        norm = np.linalg.norm(gt[28, :] - gt[26, :])
    else:
        raise ValueError('Wrong Dataset')

    error_real = np.mean(np.linalg.norm((pred - gt), axis=1) / norm)

    return error_real


class Consistency_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # HEADCAMCAL
        # BROW_INNER_LEFT: 0
        # BROW_INNER_RIGHT: 2
        # BROW_OUTER_LEFT: 14
        # BROW_OUTER_RIGHT: 15
        # NOSE_RIDGE_TIP: 72

        self.feature_inds = [0, 2, 14, 15, 72]
        # self.feature_inds = [72, ]

    def forward(self, input_tensor, ground_truth, feature_map, calibration_feature_map,
                cal_landmarks, model, stage):
        # consistency loss

        # calibration features
        ROI_feature_cal, _, _ = model.get_image_features(calibration_feature_map, cal_landmarks, stage=stage)

        # landmark features
        ROI_feature, _, _ = model.get_image_features(feature_map, input_tensor[:, -1, :, :], stage=stage)

        loss_consistency = nn.functional.mse_loss(ROI_feature[self.feature_inds, :, :, :],
                                                  ROI_feature_cal[self.feature_inds, :, :, :])

        return loss_consistency


def test():
    args = parse_args()
    update_config(cfg, args)

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # model = Sparse_alignment_network(cfg.WFLW.NUM_POINT, cfg.MODEL.OUT_DIM,
    #                                 cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
    #                                 cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
    #                                 cfg.TRANSFORMER.FEED_DIM, cfg.WFLW.INITIAL_PATH, cfg)
    model = Sparse_alignment_network_cal_refine(cfg.HEADCAMCAL.NUM_POINT, cfg.MODEL.OUT_DIM,
                                                cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                                cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                                cfg.TRANSFORMER.FEED_DIM, cfg.HEADCAMCAL.INITIAL_PATH, cfg)
    model.cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    from Dataloader.WFLW_loader import WFLWCal_Dataset
    valid_dataset = WFLWCal_Dataset(
        cfg, cfg.HEADCAMCAL.ROOT,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        annotation_file=os.path.join(cfg.HEADCAMCAL.ROOT,
                                     'HEADCAMCAL_annotations', 'list_85pt_rect_attr_train_test',
                                     'list_85pt_rect_attr_test.txt'),
        calibration_annotation_file=os.path.join(cfg.HEADCAMCAL.ROOT,
                                                 'HEADCAMCAL_annotations', 'list_85pt_rect_attr_train_test',
                                                 'list_85pt_rect_attr_calibration_test.txt'),
        wflw_config=cfg.HEADCAMCAL,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint_file = os.path.join(args.modelDir, args.checkpoint)
    checkpoint = torch.load(checkpoint_file)

    pretrained_dict = {k: v for k, v in checkpoint.items()
                       if k in model.state_dict().keys()}

    model.load_state_dict(pretrained_dict)

    model.eval()

    error_list = []
    start_error_list = []
    improvement_list = []

    consistency_loss_instance = Consistency_Loss()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)

    with torch.no_grad():
        for i, (input, input_cal, meta, meta_cal) in enumerate(valid_loader):
            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            ground_truth = meta['Points'].cuda().float()
            calibration_points = meta_cal['Points'].cuda().float()
            start_points = meta['StartPoints'].cuda().float()
            Trans = meta['trans'].numpy()[0]

            # calibration_points = calibration_points.view(1, 1, Annotated_Points.shape[0],
            #                                              Annotated_Points.shape[1]).float()
            model.calibrate(input_cal.cuda(), calibration_points)

            # start_points = start_points.view(1, 1, Annotated_Points.shape[0], Annotated_Points.shape[1]).float()
            landmarks = model(input.cuda(), start_points, input_cal, calibration_points)

            output = landmarks[2][0, -1, :, :].cpu().numpy()

            error = calcuate_loss(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)
            start_error = calcuate_loss(cfg.DATASET.DATASET, start_points[0, :, :].cpu().numpy() * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)

            feature_map = model.backbone(input.cuda())
            calibration_feature_map = model.backbone(input_cal.cuda())
            consistency_loss = consistency_loss_instance(landmarks[2], ground_truth, feature_map,
                                                         calibration_feature_map,
                                                         calibration_points, model, 3)
            start_consistency_loss = consistency_loss_instance(start_points.view(1, 1, 85, 2), ground_truth, feature_map,
                                                         calibration_feature_map,
                                                         calibration_points, model, 3)

            msg = f'Epoch: [{i}/{len(valid_loader)-1}]\t' \
                  f'NME: {error * 100.0:.3f}%\t' \
                  f'Start NME: {start_error * 100.0:.3f}%\t' \
                  f'Consistency: {consistency_loss}\t' \
                  f'Start Consistency: {start_consistency_loss}\t'

            improvement = start_error - error

            print(msg)
            error_list.append(error)
            start_error_list.append(start_error)
            improvement_list.append(improvement)

            # cal_im = input_cal.cpu().numpy().reshape([3, 256, 256]).transpose([1, 2, 0])
            # cal_im = (cal_im * 0.3) + 0.5
            # ax[0].cla()
            # ax[0].imshow(cal_im)
            # calibration_points = calibration_points.cpu().numpy() * 255
            # ax[0].scatter(calibration_points[0, [0, 2, 14, 15, 72], 0],
            #               calibration_points[0, [0, 2, 14, 15, 72], 1])
            # im = input.cpu().numpy().reshape([3, 256, 256]).transpose([1, 2, 0])
            # im = (im * 0.3) + 0.5
            # ax[1].cla()
            # ax[1].imshow(im)
            # start_points = start_points.cpu().numpy() * 255
            # ax[1].scatter(start_points[0, [0, 2, 14, 15, 72], 0],
            #               start_points[0, [0, 2, 14, 15, 72], 1])
            # output = output * 255
            # ax[1].scatter(output[[0, 2, 14, 15, 72], 0],
            #               output[[0, 2, 14, 15, 72], 1])
            # fig.show()
            # pass

        print("finished")
        print("Mean Error: {:.3f}".format((np.mean(np.array(error_list)) * 100.0)))
        print("Mean Start Error: {:.3f}".format((np.mean(np.array(start_error_list)) * 100.0)))
        print("Mean Improvement: {:.3f}".format((np.mean(np.array(improvement_list)) * 100.0)))


def test_consistency():
    args = parse_args()
    update_config(cfg, args)

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # model = Sparse_alignment_network(cfg.WFLW.NUM_POINT, cfg.MODEL.OUT_DIM,
    #                                 cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
    #                                 cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
    #                                 cfg.TRANSFORMER.FEED_DIM, cfg.WFLW.INITIAL_PATH, cfg)
    model = Sparse_alignment_network_cal_refine(cfg.HEADCAMCAL.NUM_POINT, cfg.MODEL.OUT_DIM,
                                                cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                                cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                                cfg.TRANSFORMER.FEED_DIM, cfg.HEADCAMCAL.INITIAL_PATH, cfg)
    model.cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    from Dataloader.WFLW_loader import WFLWCal_Dataset
    valid_dataset = WFLWCal_Dataset(
        cfg, cfg.HEADCAMCAL.ROOT,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        annotation_file=os.path.join(cfg.HEADCAMCAL.ROOT,
                                     'HEADCAMCAL_annotations', 'list_85pt_rect_attr_train_test',
                                     'list_85pt_rect_attr_test.txt'),
        calibration_annotation_file=os.path.join(cfg.HEADCAMCAL.ROOT,
                                                 'HEADCAMCAL_annotations', 'list_85pt_rect_attr_train_test',
                                                 'list_85pt_rect_attr_calibration_test.txt'),
        wflw_config=cfg.HEADCAMCAL,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint_file = os.path.join(args.modelDir, args.checkpoint)
    checkpoint = torch.load(checkpoint_file)

    pretrained_dict = {k: v for k, v in checkpoint.items()
                       if k in model.state_dict().keys()}

    model.load_state_dict(pretrained_dict)

    model.eval()

    error_list = []

    consistency_loss_instance = Consistency_Loss()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)

    with torch.no_grad():
        for i, (input, input_cal, meta, meta_cal) in enumerate(valid_loader):
            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            ground_truth = meta['Points'].cuda().float()
            calibration_points = meta_cal['Points'].cuda().float()
            start_points = meta['StartPoints'].cuda().float()
            Trans = meta['trans'].numpy()[0]

            feature_map = model.backbone(input.cuda())
            calibration_feature_map = model.backbone(input_cal.cuda())

            n_steps = 11
            range = 0.1
            consistency_loss = np.empty((n_steps, n_steps))
            for ii, x_off in enumerate(np.linspace(-range, range, n_steps)):
                for ij, y_off in enumerate(np.linspace(-range, range, n_steps)):
                    landmarks = ground_truth.clone()
                    landmarks[:, :, 0] += x_off
                    landmarks[:, :, 1] += y_off

                    consistency_loss[ii,ij] = consistency_loss_instance(landmarks.view(1, 1, 85, 2), ground_truth, feature_map,
                                                         calibration_feature_map,
                                                         calibration_points, model, 3)

            ax.cla()
            ax.imshow(consistency_loss)
            fig.show()
            pass
        print("finished")


if __name__ == '__main__':
    main()
    # test()
    # test_consistency()
