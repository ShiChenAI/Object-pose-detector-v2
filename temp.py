import argparse
import time
import cv2
import warnings

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, inference_bottom_up_pose_model,
                         init_pose_model, process_mmdet_results)
from mmpose.datasets import DatasetInfo
from utils import display_results, generate_obj_colors

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
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

    args = parser.parse_args()
    return args

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

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    camera = cv2.VideoCapture(args.camera_id)

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        total_tic = time.time()
        ret_val, img = camera.read()

        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2).
        mmdet_results = inference_detector(det_model, img)

        if dataset == 'TopDownCocoDataset':
            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                img,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
        elif dataset == 'BottomUpCocoDataset':
            # test a single image.
            pose_results, returned_outputs = inference_bottom_up_pose_model(pose_model,
                                                                            img,
                                                                            dataset=dataset,
                                                                            dataset_info=dataset_info,
                                                                            pose_nms_thr=args.pose_nms_thr,
                                                                            return_heatmap=return_heatmap,
                                                                            outputs=output_layer_names)           
            
        frame = display_results(img,
                                mmdet_results,
                                pose_results,
                                bbox_thr=args.bbox_thr,
                                kpt_thr=args.kpt_thr,
                                dataset=dataset,
                                dataset_info=None,
                                obj_colors=obj_colors,
                                obj_class_names=det_model.CLASSES,
                                text_color='blue',
                                radius=args.radius,
                                thickness=args.thickness)
        
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        #cv2.namedWindow('video', 0)
        cv2.imshow('video', frame)
        total_toc = time.time()
        total_time = total_toc - total_tic
        frame_rate = 1 / total_time
        print('{:.2f}fps'.format(frame_rate))
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()