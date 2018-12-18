import argparse
import PyOpenPose as OP
import cv2
import os
import numpy as np

CROP_SIZE = 256
OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]

def resize(image):
    """Crop and resize image for pix2pix."""
    height = image.shape[0]
    width = image.shape[1]
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
        return image_resize


def main():
    # OpenPose
    with_face = with_hands = True
    op = OP.OpenPose((320, 240), (240, 240), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                      False, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)

    # OpenCV
    cap = cv2.VideoCapture(args.video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output, fourcc, fps, (CROP_SIZE, CROP_SIZE))

    while cap.isOpened():
        try:
            ret, frame = cap.read()
        except Exception as e:
            print("Failed to grab", e)
            break

        if not ret or frame is None:
            break

        rgb_resize = cv2.resize(frame, (640, 480))

        op.detectPose(rgb_resize)
        op.detectFace(rgb_resize)
        op.detectHands(rgb_resize)

        res = op.render(rgb_resize)
        persons = op.getKeypoints(op.KeypointType.POSE)[0]

        if persons is not None and len(persons) > 1:
            print("First Person: ", persons[0].shape)

        gray = cv2.cvtColor(res - rgb_resize, cv2.COLOR_RGB2GRAY)
        ret, resize_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        resize_binary = cv2.cvtColor(resize_gray, cv2.COLOR_GRAY2RGB)
        out.write(resize(resize_binary))

    cap.release()
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_source', help='Device index of the camera.')
    parser.add_argument('video_output', help='Output video file.')
    args = parser.parse_args()
    main()
