
import argparse
import time
import cv2
import os
import warnings

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, inference_bottom_up_pose_model,
                         init_pose_model, process_mmdet_results)
from mmpose.datasets import DatasetInfo
from utils import display_results, generate_obj_colors, process_det_results, display_processed_results

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection images demo')
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox_thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--pose-nms-thr',
        type=float,
        default=0.9,
        help='OKS threshold for pose NMS')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument('--image-source', type=str, 
                        help='Directory of images/name of a image.')
    parser.add_argument('--save-dir', type=str, 
                        help='Save directory of output images.')
    args = parser.parse_args()
    return args


def process_img(img_path, 
                det_model, 
                dataset,
                pose_model, 
                det_cat_id,
                bbox_thr, 
                dataset_info, 
                kpt_thr, 
                obj_colors, 
                radius, 
                thickness, 
                save_dir,
                pose_nms_thr=0.5):
    
    total_tic = time.time()

    img = cv2.imread(img_path)
    croped_h = img.shape[0] - img.shape[1]
    img = img[int(croped_h):, :, :]
    #img = img[int(croped_h/2)+200: img.shape[0]-int(croped_h/2)-200, 200: img.shape[1]-200, :]  
    # get the detection results of current frame
    # the resulting box is (x1, y1, x2, y2).
    mmdet_results = inference_detector(det_model, img)
    bboxes, labels  = process_det_results(mmdet_results)
    is_malicious = 0
    if 0 in labels:
        is_malicious = 1
        if 1 in labels:
            is_malicious = 2
        elif 2 in labels:
            is_malicious = 3

    if dataset == 'TopDownCocoDataset':
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_cat_id)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None)
    elif dataset == 'BottomUpCocoDataset':
        # test a single image.
        pose_results, returned_outputs = inference_bottom_up_pose_model(pose_model,
                                                                        img,
                                                                        dataset=dataset,
                                                                        dataset_info=dataset_info,
                                                                        pose_nms_thr=pose_nms_thr,
                                                                        return_heatmap=False,
                                                                        outputs=None)           

    """    
    img_show = display_results(img,
                               mmdet_results,
                               pose_results,
                               bbox_thr=bbox_thr,
                               kpt_thr=kpt_thr,
                               dataset=dataset,
                               dataset_info=None,
                               obj_colors=obj_colors,
                               obj_class_names=det_model.CLASSES,
                               text_color='blue',
                               radius=radius,
                               thickness=thickness)
    """
    img_show = display_processed_results(img,
                                        bboxes,
                                        labels,
                                        pose_results,
                                        kpt_thr=kpt_thr,
                                        dataset=dataset,
                                        dataset_info=None,
                                        obj_colors=obj_colors,
                                        obj_class_names=det_model.CLASSES,
                                        text_color='blue',
                                        radius=radius,
                                        thickness=thickness)
    total_toc = time.time()
    total_time = total_toc - total_tic
    frame_rate = 1 / total_time
    print('{:.2f}fps'.format(frame_rate))
    
    #cv2.namedWindow('video', 0)
    if is_malicious == 2:
        malicious_str = 'Malicious!'
        sg = cv2.imread('./data/1.png')
        sg = cv2.resize(sg, img_show.shape[:2], interpolation=cv2.INTER_CUBIC)
    elif is_malicious == 3:
        malicious_str = 'Malicious!'
        sg = cv2.imread('./data/2.png')
        sg = cv2.resize(sg, img_show.shape[:2], interpolation=cv2.INTER_CUBIC)
    elif is_malicious == 1:
        malicious_str = 'Safe'
        sg = cv2.imread('./data/0.png')
        sg = cv2.resize(sg, img_show.shape[:2], interpolation=cv2.INTER_CUBIC)
    elif is_malicious == 0:
        malicious_str = 'Safe'
        sg = cv2.imread('./data/-1.png')
        sg = cv2.resize(sg, img_show.shape[:2], interpolation=cv2.INTER_CUBIC)
    #cv2.putText(img_show, '{}'.format(malicious_str), (20, 20),
    #            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 3)
    cv2.putText(img_show, '{}'.format(malicious_str), (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 7)
    img_show = cv2.hconcat([img_show, sg])
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.split(img_path)[1]
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, img_show)
        print('Save to {}'.format(save_path))
    else:
        cv2.imshow('image', img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return is_malicious
    
def main():
    args = parse_args()

    print('Initializing model...')
    # build the detection model from a config file and a checkpoint file.
    det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device.lower())

    # get object bounding box colors.
    obj_colors = generate_obj_colors(det_model.CLASSES)

    # build the pose model from a config file and a checkpoint file.
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    #pos_list = range(64, 431)
    #pos_list = range(696)
    pos_list = range(83, 490)
    tp, fp, tn, fn = 0, 0, 0, 0
    if os.path.isdir(args.image_source):
        image_list = os.listdir(args.image_source)
        image_list = sorted(image_list, 
                  key=lambda x: os.path.getmtime(os.path.join(args.image_source, x))) if image_list else []
        for image_name in image_list:
            image_path = os.path.join(args.image_source, image_name)
            img_id = int(os.path.splitext(image_name)[0].split('_')[-1])
            is_malicious = process_img(image_path,
                                        det_model,
                                        dataset,
                                        pose_model,
                                        args.det_cat_id,
                                        args.bbox_thr,
                                        dataset_info,
                                        args.kpt_thr,
                                        obj_colors,
                                        args.radius,
                                        args.thickness,
                                        args.save_dir)
            print(img_id, is_malicious)
            if img_id in pos_list:
                if is_malicious in [2, 3]:
                    tp += 1
                else:
                    fn += 1
            else:
                if is_malicious in [2, 3]:
                    fp += 1
                else:
                    tn += 1
        precison = tp / (tp + fp)
        recall = tp / (tp + fn)
        print('tp: {}'.format(tp))
        print('fp: {}'.format(fp))
        print('fn: {}'.format(fn))
        print('tn: {}'.format(tn))
        print('precision: {}'.format(precison))
        print('recall: {}'.format(recall))

    elif os.path.isfile(args.image_source):     
        process_img(args.image_source,
                    det_model,
                    dataset,
                    pose_model,
                    args.det_cat_id,
                    args.bbox_thr,
                    dataset_info,
                    args.kpt_thr,
                    obj_colors,
                    args.radius,
                    args.thickness,
                    args.save_dir)

if __name__ == '__main__':
    main()