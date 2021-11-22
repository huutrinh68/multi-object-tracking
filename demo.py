import numpy as np
import collections
import cv2
from motrackers.detectors import TF_SSDMobileNetV2
from motrackers import CentroidTracker
from motrackers.utils import draw_tracks
# import time

VIDEO_FILE = "./examples/video_data/Aeon_inout_movie2_20fps.mp4"
WEIGHTS_PATH = ('./examples/pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb')
CONFIG_FILE_PATH = './examples/pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
LABELS_PATH = "./examples/pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_names.json"

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.2
DRAW_BOUNDING_BOXES = True
USE_GPU = False


tracker = CentroidTracker(max_lost=2, tracker_output_format='mot_challenge')


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


def checkLeft(pts, x_c):
    global H, W
    if len(pts) >= 2 and pts[-1] < W // 2 and pts[-2] >= W // 2:
        direction = x_c - np.mean(pts)
        if direction < 0:
            return True
        else:
            return False
    else:
        return False


def checkRight(pts, x_c):
    global H, W
    if len(pts) >= 2 and pts[-1] > W // 2 and pts[-2] <= W // 2:
        direction = x_c - np.mean(pts)
        if direction > 0:
            return True
        else:
            return False
    else:
        return False


def main(video_path, model, tracker):
    global H, W
    totalLeft = 0
    totalRight = 0
    counted_ids = []
    pts = collections.defaultdict(list)

    if True:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter('output.mp4', fourcc, 30, (640, 480), True)

    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    cnt = 0
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

        # draw boundingbox
        image = model.draw_bboxes(image, bboxes, confidences, class_ids)

        # draw centertroid
        image = draw_tracks(image, tracks)

        # update monitor result
        for track in tracks:
            # track_id
            track_id = track[1]

            # x coordinate
            x_coord = track[2]

            # width
            width = track[4]

            # x_center
            x_c = int(x_coord + 0.5 * width)

            # append x coordinate
            pts[track_id].append(x_c)

            if track_id not in counted_ids and checkRight(pts[track_id], x_c):
                counted_ids.append(track_id)
                totalRight += 1
                del pts[track_id]
            if track_id not in counted_ids and checkLeft(pts[track_id], x_c):
                counted_ids.append(track_id)
                totalLeft += 1
                del pts[track_id]

        # draw center line
        cv2.line(image, (W // 2, 0), (W // 2, H), (0, 0, 255), 2)

        # construct a tuple of information we will be displaying on the
        info = [
            ("in", totalLeft),
            ("out", totalRight),
        ]

        # Display the monitor result
        for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(image, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if writer is not None:
            writer.write(image)

        # show result
        cv2.imshow("People Counter", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # time.sleep(0.01)
        cnt += 1
        if cnt == 1000:
            cnt = 0

    cap.release()
    cv2.destroyAllWindows()


main(VIDEO_FILE, model, tracker)
