import argparse
import cv2
import os
import numpy as np
import tensorflow as tf

CROP_SIZE = 256

def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory."""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def main():
    # TensorFlow
    graph = load_graph(args.frozen_model_file)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)

    # determine image size automatically from trained model
    CROP_SIZE = int(image_tensor.shape[0])
    print("CROP_SIZE:", CROP_SIZE)

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

        # generate prediction
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
        image_rgb = np.concatenate([image_rgb, image_rgb], axis=1)
        generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
        image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
        out.write(image_bgr)

    sess.close()
    cap.release()
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_source', help='Device index of the camera.')
    parser.add_argument('video_output', help='Output video file.')
    parser.add_argument('frozen_model_file', help='Frozen TensorFlow model file.')
    args = parser.parse_args()
    main()
