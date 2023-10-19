import argparse
import os.path

from Config import cfg
from Config import update_config

from Backbone import get_face_alignment_net

from utils import create_logger
from SLPT import Sparse_alignment_network
from Dataloader import WFLW_test_Dataset
from SLPT.Transformer import Transformer

import torch, cv2, math
import numpy as np
import pprint
from torch import nn

import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import Face_Detector
import utils

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
        self.calibration_encoding = nn.Parameter(torch.randn(1, num_points, self.d_model))

        # SLPT_Inherent_Layer = Inherent_Layer(d_model, nhead, dim_feedforward, dropout,
        #                                             activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.Transformer_block = Transformer_block(SLPT_Inherent_Layer, num_decoder_layer, decoder_norm, return_intermediate=True)

        self._reset_parameters()

    def forward(self, src, cal):
        bs, num_feat, len_feat = src.size()

        structure_encoding = self.structure_encoding.repeat(bs, 1, 1).permute(1, 0, 2)
        calibration_encoding = self.calibration_encoding.repeat(bs, 1, 1).permute(1, 0, 2)
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


class Sparse_alignment_network_cal_refine(Sparse_alignment_network):
    def __init__(self, num_point, d_model, trainable,
                 return_interm_layers, dilation, nhead,  feedforward_dim,
                 initial_path, cfg):
        super().__init__(num_point, d_model, trainable,
                 return_interm_layers, dilation, nhead,  feedforward_dim,
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
        ROI_anchor_cal_2, bbox_size_cal_2, start_anchor_cal_2 = self.ROI_2(cal_landmarks[:, -1, :, :].detach())
        ROI_anchor_cal_2 = ROI_anchor_cal_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_cal_2 = self.interpolation(calibration_feature_map, ROI_anchor_cal_2.detach()).view(bs, self.num_point, self.Sample_num,
                                                                            self.Sample_num, self.d_model)
        ROI_feature_cal_2 = ROI_feature_cal_2.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                     self.d_model).permute(0, 3, 2, 1)

        transformer_feature_cal_2 = self.feature_extractor_cal(ROI_feature_cal_2).view(bs, self.num_point, self.d_model)

        ROI_anchor_cal_3, bbox_size_cal_3, start_anchor_cal_3 = self.ROI_3(cal_landmarks[:, -1, :, :].detach())
        ROI_anchor_cal_3 = ROI_anchor_cal_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_cal_3 = self.interpolation(calibration_feature_map, ROI_anchor_cal_3.detach()).view(bs, self.num_point, self.Sample_num,
                                                                            self.Sample_num, self.d_model)
        ROI_feature_cal_3 = ROI_feature_cal_3.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                     self.d_model).permute(0, 3, 2, 1)
        transformer_feature_cal_3 = self.feature_extractor_cal(ROI_feature_cal_3).view(bs, self.num_point, self.d_model)

        self.transformer_feature_cal_2 = transformer_feature_cal_2
        self.transformer_feature_cal_3 = transformer_feature_cal_3

    def forward(self, image, landmarks_1=None, landmarks_2=None ):

        bs = image.size(0)

        output_list = []

        feature_map = self.backbone(image)

        initial_landmarks = self.initial_points.repeat(bs, 1, 1).to(image.device)

        # stage_2
        ROI_anchor_2, bbox_size_2, start_anchor_2 = self.ROI_2(landmarks_1[:, -1, :, :].detach())
        ROI_anchor_2 = ROI_anchor_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_2 = self.interpolation(feature_map, ROI_anchor_2.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                 self.Sample_num, self.d_model)
        ROI_feature_2 = ROI_feature_2.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                           self.d_model).permute(0, 3, 2, 1)

        transformer_feature_2 = self.feature_extractor(ROI_feature_2).view(bs, self.num_point, self.d_model)

        offset_2 = self.Transformer(transformer_feature_2, self.transformer_feature_cal_2)
        offset_2 = self.out_layer(offset_2)

        landmarks_2 = start_anchor_2.unsqueeze(1) + bbox_size_2.unsqueeze(1) * offset_2
        output_list.append(landmarks_2)

        # stage_3
        ROI_anchor_3, bbox_size_3, start_anchor_3 = self.ROI_3(landmarks_2[:, -1, :, :].detach())
        ROI_anchor_3 = ROI_anchor_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_3= self.interpolation(feature_map, ROI_anchor_3.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                   self.Sample_num, self.d_model)
        ROI_feature_3 = ROI_feature_3.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                           self.d_model).permute(0, 3, 2, 1)

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


def run_refinement(image_files, image_landmarks, cfg, normalize, model):
    redo_track = True
    display = False

    output_dir = r'C:\temp\SLPT\TestData\SLPT'
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
                       cal_image_file, cal_image_landamrks, cfg, normalize, model):
    redo_track = True
    display = False

    output_dir = r'C:\temp\SLPT\TestData\SLPTCal'
    refined_landmarks = image_landmarks
    output_file = os.path.basename(os.path.dirname(image_files[0]))
    print(f'Processing {output_file}')
    output_file = os.path.join(output_dir, f'{output_file}.npz')

    if not redo_track and os.path.exists(output_file):
        return

    frame = cv2.imread(cal_image_file)

    bbox = get_bbox(cal_image_landamrks, cfg)
    cal_input, trans = crop_img(frame.copy(), bbox, normalize)

    cal_landmarks_model = torch.from_numpy(
        utils.transform_pixel_v2(cal_image_landamrks, trans) / cfg.MODEL.IMG_SIZE
    ).view(1, 1, cal_image_landamrks.shape[0], cal_image_landamrks.shape[1]).float()

    model.module.calibrate(cal_input.cuda(), cal_landmarks_model.cuda())

    import tqdm
    for i, (image_file, landmarks) in tqdm.tqdm(enumerate(zip(image_files, image_landmarks)), total=len(image_files)):
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


def run_frame_to_frame(image_files, image_landmarks, cfg, normalize, model):
    redo_track = False
    display = False

    output_dir = r'C:\temp\SLPT\TestData\Update'
    refined_landmarks = image_landmarks
    output_file = os.path.basename(os.path.dirname(image_files[0]))
    print(f'Processing {output_file}')
    output_file = os.path.join(output_dir, f'{output_file}.npz')

    if not redo_track and os.path.exists(output_file):
        return

    import tqdm
    prev_landmarks = image_landmarks[0]
    for i, image_file in tqdm.tqdm(enumerate(image_files)):
        frame = cv2.imread(image_file)

        bbox = get_bbox(prev_landmarks, cfg)
        alignment_input, trans = crop_img(frame.copy(), bbox, normalize)

        landmarks_model = torch.from_numpy(
            utils.transform_pixel_v2(prev_landmarks, trans) / cfg.MODEL.IMG_SIZE
        ).view(1, 1, prev_landmarks.shape[0], prev_landmarks.shape[1]).float()

        outputs_initial = model(alignment_input.cuda(),
                                landmarks_1=landmarks_model.cuda(),
                                landmarks_2=None,
                                )
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
    method = 'refinement_cal'  # refinement, detection, frame_to_frame

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

    if method in ['refinement', 'frame_to_frame']:
        model = Sparse_alignment_network_refine(cfg.HEADCAM.NUM_POINT, cfg.MODEL.OUT_DIM,
                                                cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                                cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                                cfg.TRANSFORMER.FEED_DIM, cfg.HEADCAM.INITIAL_PATH, cfg)
    elif method == 'refinement_cal':
        model = Sparse_alignment_network_cal_refine(cfg.HEADCAMCAL.NUM_POINT, cfg.MODEL.OUT_DIM,
                                                cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                                cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                                cfg.TRANSFORMER.FEED_DIM, cfg.HEADCAMCAL.INITIAL_PATH, cfg)
    else:
        model = Sparse_alignment_network(cfg.HEADCAM.NUM_POINT, cfg.MODEL.OUT_DIM,
                                         cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                         cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                         cfg.TRANSFORMER.FEED_DIM, cfg.HEADCAM.INITIAL_PATH, cfg)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    checkpoint_file = os.path.join(args.modelDir, args.checkpoint)
    checkpoint = torch.load(checkpoint_file)
    pretrained_dict = {k: v for k, v in checkpoint.items()
                       if k in model.module.state_dict().keys()}
    model.module.load_state_dict(pretrained_dict)
    model.eval()

    print('Finished loading face landmark detector')

    # test data file
    import glob
    test_files = glob.glob(r'C:\temp\SLPT\TestData\*.npz')

    for test_data_file in test_files:
        # test_data_file = r"C:\temp\SLPT\TestData\video_2023-06-05_17-26-06_Frames.npz"
        # test_data_file = r"C:\temp\SLPT\TestData\Alfonso_l_sc04_001_39_1_Frames.npz"
        # test_data_file = r"C:\temp\SLPT\TestData\FaceCapture_Catt_Act6.1Scene1_Frames.npz"

        npz_file = np.load(test_data_file)
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
            run_refinement(image_files, landmarks, cfg, normalize, model)
        elif method == 'frame_to_frame':
            run_frame_to_frame(image_files, landmarks, cfg, normalize, model)
        elif method == 'detector':
            run_with_detector(image_files, cfg, net, normalize, model)
        elif method == 'refinement_cal':
            video_name = os.path.basename(test_data_file)[:-11]
            cal_ind = _CALIBRATION_FRAMES[video_name]
            cal_image_file = image_files[cal_ind]
            cal_image_landmarks = landmarks[cal_ind, :, :]
            run_refinement_cal(image_files, landmarks, cal_image_file, cal_image_landmarks, cfg, normalize, model)
        else:
            raise RuntimeError('Unknown method')


if __name__ == '__main__':
    main()
