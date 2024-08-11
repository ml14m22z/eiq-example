import argparse
import tflite_runtime.interpreter as tflite
import numpy as np
import pathlib
import cv2
from PIL import Image
import typing

MODEL_PATH = pathlib.Path("../models/")
DETECT_MODEL = "face_detection_front_128_full_integer_quant.tflite"
LANDMARK_MODEL = "face_landmark_192_integer_quant.tflite"
EYE_MODEL = "iris_landmark_quant.tflite"

SCORE_THRESH = 0.9
MAX_FACE_NUM = 1
ANCHOR_STRIDES = [8, 16]
ANCHOR_NUM = [2, 6]


def create_anchors(input_shape):
    w, h = input_shape
    anchors = []
    for s, a_num in zip(ANCHOR_STRIDES, ANCHOR_NUM):
        gridCols = (w + s - 1) // s
        gridRows = (h + s - 1) // s
        x, y = np.meshgrid(np.arange(gridRows), np.arange(gridCols))
        print(f'x.shape: {x.shape}, y.shape: {y.shape}')
        x, y = x[..., None], y[..., None]
        print(f'x.shape: {x.shape}, y.shape: {y.shape}')
        anchor_grid = np.concatenate([y, x], axis=-1)
        print(f'anchor_grid.shape: {anchor_grid.shape}')
        anchor_grid = np.tile(anchor_grid, (1, 1, a_num))
        anchor_grid = s * (anchor_grid.reshape(-1, 2) + 0.5)
        anchors.append(anchor_grid)
        print(f'anchor_grid.shape: {anchor_grid.shape}')
        # print(f'anchor_grid: {anchor_grid}')
    return np.concatenate(anchors, axis=0)


def decode(scores, bboxes):
    w, h = input_shape
    top_score = np.sort(scores)[-MAX_FACE_NUM]
    cls_mask = scores >= max(SCORE_THRESH, top_score)
    if cls_mask.sum() == 0:
        return np.array([]), np.array([]), np.array([])

    scores = scores[cls_mask]
    bboxes = bboxes[cls_mask]
    bboxes_anchors = anchors[cls_mask]

    bboxes_decoded = bboxes_anchors.copy()
    bboxes_decoded[:, 0] += bboxes[:, 1]  # row
    bboxes_decoded[:, 1] += bboxes[:, 0]  # columns
    bboxes_decoded[:, 0] /= h
    bboxes_decoded[:, 1] /= w

    pred_w = bboxes[:, 2] / w
    pred_h = bboxes[:, 3] / h

    topleft_x = bboxes_decoded[:, 1] - pred_w * 0.5
    topleft_y = bboxes_decoded[:, 0] - pred_h * 0.5
    btmright_x = bboxes_decoded[:, 1] + pred_w * 0.5
    btmright_y = bboxes_decoded[:, 0] + pred_h * 0.5

    pred_bbox = np.stack([topleft_x, topleft_y, btmright_x, btmright_y], axis=-1)

    # decode landmarks
    landmarks = bboxes[:, 4:]
    landmarks[:, 1::2] += bboxes_anchors[:, 0:1]
    landmarks[:, ::2] += bboxes_anchors[:, 1:2]
    landmarks[:, 1::2] /= h
    landmarks[:, ::2] /= w

    return pred_bbox, landmarks, scores


def nms_oneclass(bbox, score, thresh = 0.4):
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def resize_crop_image(
        original_image: Image.Image,
        image_size: typing.Sequence
) -> np.ndarray:
    """
    Resize and crop input image

    @param original_image:  Image to resize and crop
    @param image_size:      New image size
    @return:                Resized and cropped image
    """
    # IFM size
    ifm_width = image_size[0]
    ifm_height = image_size[1]

    # Aspect ratio resize
    scale_ratio = (float(max(ifm_width, ifm_height))
                   / float(min(original_image.size[0], original_image.size[1])))
    resized_width = int(original_image.size[0] * scale_ratio)
    resized_height = int(original_image.size[1] * scale_ratio)
    resized_image = original_image.resize(
        size=(resized_width, resized_height),
        resample=Image.Resampling.BILINEAR
    )

    # Crop the center of the image
    resized_image = resized_image.crop((
        (resized_width - ifm_width) / 2,  # left
        (resized_height - ifm_height) / 2,  # top
        (resized_width + ifm_width) / 2,  # right
        (resized_height + ifm_height) / 2  # bottom
    ))

    return np.array(resized_image, dtype=np.uint8).flatten()


def draw_face_box(image, bboxes, landmarks, scores):
    for bbox, landmark, score in zip(bboxes.astype(int), landmarks.astype(int), scores):
        # print(f'{bbox}, {landmark}, {score} = zip(bboxes.astype(int), landmarks.astype(int), scores)')
        image = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color=(255, 0, 0), thickness=2)
        landmark = landmark.reshape(-1, 2)
        # print(f'{landmark} = landmark.reshape(-1, 2)')
        # draw face landmarks
        # for i, l in enumerate(landmark):
        #     cv2.circle(image, tuple(l), 2, (0, 255, 0), thickness=2)
        #     cv2.putText(image, str(i), (tuple(l)[0] + 5, tuple(l)[1] + 5),
        #         cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2)

        score_label = "{:.2f}".format(score)
        (label_width, label_height), baseline = cv2.getTextSize(score_label,
                                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                                fontScale=1.0,
                                                                thickness=2)
        label_btmleft = bbox[:2].copy() + 10
        label_btmleft[0] += label_width
        label_btmleft[1] += label_height
        cv2.rectangle(image, tuple(bbox[:2]), tuple(label_btmleft), color=(255, 0, 0), thickness=cv2.FILLED)
        cv2.putText(image, score_label, (bbox[0] + 5, label_btmleft[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='/dev/video0', help='input to be classified')
    args = parser.parse_args()

    # gen_rgb_cpp.py
    original_image = Image.open(args.input).convert("RGB")
    # print('original_image.size:', original_image.size)
    # print('original_image:')
    # for row in range(3):
    #     for col in range(3):
    #         print(original_image.getpixel((row, col)), end=' ')
    #     print('')

    resized_image = resize_crop_image(original_image, (128, 128))
    rgb_data = resized_image.reshape(128, 128, 3)
    print('rgb_data.shape:', rgb_data.shape)
    print('rgb_data:')
    for row in range(3):
        for col in range(3):
            print(rgb_data[row, col], end=' ')
        print('')

    bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
    print('bgr_data.shape:', bgr_data.shape)
    print('bgr_data:')
    for row in range(3):
        for col in range(3):
            print(bgr_data[row, col], end=' ')
        print('')
    cv2.imshow('bgr_data', bgr_data)

    # interpreter setup
    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH / DETECT_MODEL))
    interpreter.allocate_tensors()
    input_idx = interpreter.get_input_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape'][1:3]
    outputs_idx = {}
    for output in interpreter.get_output_details():
        outputs_idx[output['name']] = output['index']
    anchors = create_anchors(input_shape)
    # print('anchors:', anchors)
    # np.savetxt('anchors.txt', anchors)
    # print('anchors.shape:', anchors.shape)

    # convert to float32
    input_data = cv2.resize(rgb_data, tuple(input_shape)).astype(np.float32)
    print('input_data.shape:', input_data.shape)
    print('input_data:')
    for row in range(3):
        for col in range(3):
            print(f'{input_data[row, col]}', end=' ')
        print('')

    input_data = (input_data - 128.0) / 128.0
    print('input_data.shape:', input_data.shape)
    print('input_data:')
    for row in range(3):
        for col in range(3):
            print(f'{input_data[row, col]}', end=' ')
        print('')

    input_data = np.expand_dims(input_data, axis=0)

    tmp = input_data.astype(np.float32)[0] * 128.0 + 128.0
    tmp = tmp.astype(np.uint8)

    interpreter.set_tensor(input_idx, input_data)
    interpreter.invoke()
    scores = interpreter.get_tensor(outputs_idx['classificators']).squeeze()
    scores = 1 / (1 + np.exp(-scores))
    bboxes = interpreter.get_tensor(outputs_idx['regressors']).squeeze()
    print('scores.shape:', scores.shape)
    print('bboxes.shape:', bboxes.shape)

    bboxes_decoded, landmarks, scores = decode(scores, bboxes)
    bboxes_decoded *= rgb_data.shape[0]
    landmarks *= rgb_data.shape[0]
    
    if len(bboxes_decoded) != 0:
        keep_mask = nms_oneclass(bboxes_decoded, scores)  # np.ones(pred_bbox.shape[0]).astype(bool)
        bboxes_decoded = bboxes_decoded[keep_mask]
        landmarks = landmarks[keep_mask]
        scores = scores[keep_mask]

    print('rgb_data.shape:', rgb_data.shape)
    print('input_shape:', input_shape)
    print('input_data.shape:', input_data.shape)
    print('input_data:', input_data[0][0][0:10])
    print('tmp.shape:', tmp.shape)
    print('keep_mask:', keep_mask)
    print('bboxes_decoded:', bboxes_decoded)
    print('landmarks:', landmarks)
    print('scores:', scores)

    cv2.imshow('image', cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR))
    cv2.imshow('resized', cv2.cvtColor(rgb_data.reshape(128, 128, 3), cv2.COLOR_RGB2BGR))
    # draw
    draw_face_box(tmp, bboxes_decoded, landmarks, scores)
    cv2.imshow('input', cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break