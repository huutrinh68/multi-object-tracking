import numpy as np
import collections
import cv2
from motrackers.detectors import TF_SSDMobileNetV2, YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks
import time
import numpy
import datetime

# VIDEO_FILE = "./examples/video_data/Aeon_inout_movie3_20fps.mp4"
VIDEO_FILE = "./examples/video_data/Aeon_inout_movie2_20fps.mp4"
# VIDEO_FILE = "./examples/video_data/video2_20fps.mp4"
WEIGHTS_PATH = ('./examples/pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb')
CONFIG_FILE_PATH = './examples/pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
LABELS_PATH = "./examples/pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_names.json"

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.2
DRAW_BOUNDING_BOXES = True
USE_GPU = False


tracker = CentroidTracker(max_lost=2, tracker_output_format='mot_challenge')
# tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
# tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
# tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
#                      tracker_output_format='mot_challenge')


model = TF_SSDMobileNetV2(
    weights_path=WEIGHTS_PATH,
    configfile_path=CONFIG_FILE_PATH,
    labels_path=LABELS_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    nms_threshold=NMS_THRESHOLD,
    draw_bboxes=DRAW_BOUNDING_BOXES,
    use_gpu=USE_GPU
)

H, W = 0, 0
margin = 0
direction_lr = 1

'''
{
'0': [{'timestamp': time, 'x_c': int, 'y_c': int}, {'timestamp': time, 'x_c': int, 'y_c': int}, ...],
'1': [{'timestamp': time, 'x_c': int, 'y_c': int}, {'timestamp': time, 'x_c': int, 'y_c': int}, ...],
}
'''


# right or bottom
def checkDirection1(pts, x_c, y_c, direction):
    # check right
    if direction:
        condition = len(pts) >= 2 and pts[-1]['x_c'] > W // 2 + margin and pts[-2]['x_c'] <= W // 2 + margin
        mean_pts = np.mean([p['x_c'] for p in pts])
        move = x_c - mean_pts
    # check bottom
    else:
        condition = len(pts) >= 2 and pts[-1]['y_c'] > H // 2 + margin and pts[-2]['y_c'] <= H // 2 + margin
        mean_pts = np.mean([p['y_c'] for p in pts])
        move = y_c - mean_pts

    if condition and move > 0:
        return True
    else:
        return False


# left or top
def checkDirection2(pts, x_c, y_c, direction):
    # check left
    if direction:
        condition = len(pts) >= 2 and pts[-1]['x_c'] < W // 2 - margin and pts[-2]['x_c'] >= W // 2 - margin
        mean_pts = np.mean([p['x_c'] for p in pts])
        move = x_c - mean_pts
    # check top
    else:
        condition = len(pts) >= 2 and pts[-1]['y_c'] < H // 2 - margin and pts[-2]['y_c'] >= H // 2 - margin
        mean_pts = np.mean([p['y_c'] for p in pts])
        move = y_c - mean_pts

    if condition and move < 0:
        return True
    else:
        return False


def del_id(pts):
    previous_time = datetime.datetime.now() - datetime.timedelta(hours=0, minutes=5)
    del_ids = []
    for track_id, value in pts.items():
        # check timestamp of the first track_id
        # delete that track_id's value if timestamp is 5 minutes before
        if value[0]['timestamp'] < previous_time:
            del_ids.append(track_id)

    for del_id in del_ids:
        del pts[del_id]
    return pts


def find_stay_time(track_id, pts):
    stay_time = 0
    for key, value in pts.items():
        if track_id == key:
            stay_time = value[-1]['timestamp'] - value[0]['timestamp']
            stay_time = stay_time.total_seconds()

    return stay_time


def main(video_path, model, tracker):
    global H, W
    direction_count_1 = 0
    direction_count_2 = 0
    counted_ids = []
    pts = collections.defaultdict(list)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter('output.mp4', fourcc, 30, (640, 480), True)

    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    while True:
        ok, image = cap.read()

        if not ok:
            break

        image = cv2.resize(image, (640, 480))
        H, W = image.shape[:2]

        # people detection
        bboxes, confidences, class_ids = model.detect(image)

        # update tracker
        tracks = tracker.update(bboxes, confidences, class_ids)

        # update monitor result
        for track in tracks:
            # track_id
            track_id = track[1]
            # x coordinate
            x_coord = track[2]
            # y coordinate
            y_coord = track[3]
            # width
            width = track[4]
            # height
            height = track[5]
            # x_center
            x_c = int(x_coord + 0.5 * width)
            # y_center
            y_c = int(y_coord + 0.5 * height)

            # append x coordinate
            pts[track_id].append({'timestamp': datetime.datetime.now(), 'x_c': x_c, 'y_c': y_c})

            # make sure mean of x, y coordinate is stable
            if track_id not in counted_ids and checkDirection1(pts[track_id], x_c, y_c, direction_lr):
                counted_ids.append(track_id)
                direction_count_1 += 1
                # detected = True
                # del pts[track_id]
            if track_id not in counted_ids and checkDirection2(pts[track_id], x_c, y_c, direction_lr):
                counted_ids.append(track_id)
                direction_count_2 += 1
                # detected = True
                # del pts[track_id]

        # draw boundingbox
        # updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)
        updated_image = image.copy()
        for bb, conf, cid, track in zip(bboxes, confidences, class_ids, tracks):
            track_id = track[1]

            stay_time = find_stay_time(track_id, pts)

            # draw
            clr = (255, 0, 255)
            cv2.rectangle(updated_image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)
            label = "ID{}: {:.1f}s".format(track_id, stay_time)
            (label_width, label_height), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(bb[1], label_height)
            cv2.rectangle(updated_image, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(updated_image, label, (bb[0], y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

        # draw centertroid
        # updated_image = draw_tracks(updated_image, tracks)

        # draw center line
        # if direction_lr:
        #     cv2.line(updated_image, (W // 2, 0), (W // 2, H), (0, 0, 255), 2)
        # else:
        #     cv2.line(updated_image, (0, H // 2), (W, H // 2), (0, 0, 255), 2)

        # check time to remove not counted id
        pts = del_id(pts)
        # construct a tuple of information we will be displaying on the
        # info = [
        #     ("Left->Right", direction_count_1),
        #     ("Left<-Right", direction_count_2),
        # ]

        # Display the monitor result
        # for (i, (k, v)) in enumerate(info):
        #         text = "{}: {}".format(k, v)
        #         cv2.putText(updated_image, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if writer is not None:
            writer.write(updated_image)

        # show result
        cv2.imshow("People Counter", updated_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)
        # time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()


main(VIDEO_FILE, model, tracker)
