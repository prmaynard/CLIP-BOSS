# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
import io
import os
import sys
import time
import logging
import cv2
import shutil
from PIL import Image
import numpy as np
from bosdyn.api import network_compute_bridge_service_pb2_grpc
from bosdyn.api import network_compute_bridge_pb2
from bosdyn.api import image_pb2
from bosdyn.api import header_pb2
import bosdyn.client
import bosdyn.client.util
import grpc
from concurrent import futures

import queue
import threading
from google.protobuf import wrappers_pb2

kServiceAuthority = "fetch-tutorial-worker.spot.robot"
'''
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
'''
import os
import cv2
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import numpy as np
from pkg_resources import packaging
import clip
import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as T

import random
random.seed(5)

from collections import Counter

from sklearn.metrics import accuracy_score

from collections import OrderedDict
import torch

print("Torch version:", torch.__version__)


import os
import cv2
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def load_object_detection_model():
    ob_model = fasterrcnn_resnet50_fpn(pretrained=True)
    ob_model.eval()
    return ob_model


def detect_objects(image, model, threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    scores = outputs[0]['scores']
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']

    keep = scores > threshold
    boxes = boxes[keep]
    labels = labels[keep]

    return boxes, labels


def save_subimages(image, boxes):
    saved_bboxes = []
    subimages = []

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box

        xmin = int(xmin.item())
        ymin = int(ymin.item())
        xmax = int(xmax.item())
        ymax = int(ymax.item())


        saved_bboxes.append([xmin, ymin, xmax, ymax])
        subimage = image.crop([xmin, ymin, xmax, ymax])

        # saved_bboxes.append([ymin, xmin, ymax, xmax])
        # subimage = image.crop([ymin, xmin, ymax, xmax])
        subimages.append(subimage)

    return saved_bboxes, subimages


def get_clip_model():
    clip.available_models()

    model, preprocess = clip.load("RN50")
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    return model, preprocess


def classify_with_clip(image_file, sub_image_dir, object_detection_model, CLIP_model, CLIP_preprocessing, class_dict):
    model = object_detection_model

    image_path = image_file
    image = Image.open(image_path)

    boxes, labels = detect_objects(image, model)
    saved_bboxes, subimages = save_subimages(image, boxes)

    return_dict = {}

    print(labels)

    for i, subimage in enumerate(subimages):
        return_dict[f'subimage_{i}'] = {}
        return_dict[f'subimage_{i}']['bounding_box'] = saved_bboxes[i]
        subimage_path = f'subimage_{i}.jpg'
        subimage.save(os.path.join(sub_image_dir, subimage_path))

    print("Saved bounding boxes:", saved_bboxes)

    CLASSES = list(class_dict.keys())

    model = CLIP_model
    preprocess = CLIP_preprocessing

    text_descriptions = [f"This is a photo of a {label}" for label in CLASSES]
    text_tokens = clip.tokenize(text_descriptions).cuda()

    top_n = 1

    item_list = []

    with torch.no_grad():

        image_features_8K = []
        for item in os.listdir(sub_image_dir):
            print(item)
            item_list.append(item)
            item = Image.open(os.path.join(sub_image_dir, item)).convert("RGB")
            image = preprocess(item)

            image_feature = model.encode_image(image.unsqueeze(0).cuda()).float()
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            image_features_8K.append(image_feature)

        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        y_pred_8k = []

        for j, image_feature in enumerate(image_features_8K):
            print(j)
            text_probs = (100.0 * image_feature @ text_features.T).softmax(dim=-1)
            top_probs, top_labels = text_probs.cpu().topk(top_n, dim=-1)
            return_dict[f'subimage_{j}']['top_probs'] = text_probs.max().cpu()

            y_pred_8k.append(top_labels)

        y_pred_8k_text_labels = [CLASSES[int(y_pred_8k[i][0][0].cpu())] for i in range(len(y_pred_8k))]

        for j, image_feature in enumerate(y_pred_8k_text_labels):
            return_dict[f'subimage_{j}']['labels'] = y_pred_8k_text_labels[j]

    shutil.rmtree('.\imgs')
    os.mkdir(os.path.join('.\imgs'))

    return return_dict

    


class ObjectDetectionModel:
    def __init__(self):
        self.object_detection_model = load_object_detection_model()
        self.clip_model, self.clip_preprocessing = get_clip_model()

        self.image_path = '.\img.jpg'
        self.sub_image_dir = '.\imgs'

        if not os.path.exists(os.path.join(self.sub_image_dir)):
            os.mkdir(os.path.join(self.sub_image_dir))

    def predict(self, image):
        cv2.imwrite(self.image_path, image)


        class_dict = {'water bottle': 0, 'swirly ball': 1, 'lemon': 2}

        detections = classify_with_clip(self.image_path, self.sub_image_dir, self.object_detection_model, self.clip_model, self.clip_preprocessing, class_dict)
        
        for subimage in detections.keys():
            box = detections[subimage]['bounding_box']
            p1 = box[0]
            p2 = box[1]
            p3 = box[2]
            p4 = box[3]
            detections[subimage]['bounding_box'] = [p2, p1, p4, p3]

        return detections

def process_thread(args, request_queue, response_queue):
    # Load the model(s)
    # models = {}
    # for model in args.model:
    model = ObjectDetectionModel()
        # models['Our_model'] = this_model

    # print('')
    print('Service ' + args.name + ' running on port: ' + str(args.port))

    # print('Loaded models:')
    # for model_name in models:
        # print('    ' + model_name)

    while True:
        request = request_queue.get()
        

        if isinstance(request, network_compute_bridge_pb2.ListAvailableModelsRequest):
            out_proto = network_compute_bridge_pb2.ListAvailableModelsResponse()
            out_proto.available_models.append('clip_classifier')
            response_queue.put(out_proto)
            continue
        else:
            out_proto = network_compute_bridge_pb2.NetworkComputeResponse()

        # Find the model
        # if request.input_data.model_name not in models:
        #     err_str = 'Cannot find model "' + request.input_data.model_name + '" in loaded models.'
        #     print(err_str)

        #      # Set the error in the header.
        #     out_proto.header.error.code = header_pb2.CommonError.CODE_INVALID_REQUEST
        #     out_proto.header.error.message = err_str
        #     response_queue.put(out_proto)
        #     continue

        # model = models[request.input_data.model_name]

        # Unpack the incoming image.
        if request.input_data.image.format == image_pb2.Image.FORMAT_RAW:
            pil_image = Image.open(io.BytesIO(request.input_data.image.data))
                # If the input image is grayscale, convert it to RGB.
                # image = cv2.cvtColor(pil_image, cv2.COLOR_GRAY2RGB)

            # elif request.input_data.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                # Already an RGB image.
            image = pil_image


        elif request.input_data.image.format == image_pb2.Image.FORMAT_JPEG:
            dtype = np.uint8
            jpg = np.frombuffer(request.input_data.image.data, dtype=dtype)
            image = cv2.imdecode(jpg, -1)

            if len(image.shape) < 3:
                # If the input image is grayscale, convert it to RGB.
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_width = image.shape[0]
        image_height = image.shape[1]


        detections = model.predict(image)
    
        num_objects = 0

        # # All outputs are batches of tensors.
        # # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # # We're only interested in the first num_detections.
        # num_detections = int(detections.pop('num_detections'))
        # detections = {key: value[0, :num_detections].numpy()
        #                for key, value in detections.items()}

        # boxes = detections['detection_boxes']
        # classes = detections['detection_classes']
        # scores = detections['detection_scores']


        for subimage in detections.keys():

            prob = detections[subimage]['top_probs'].item()
            box = detections[subimage]['bounding_box']
            label = detections[subimage]['labels']

            # if prob < request.input_data.min_confidence:
            #     continue

            # box = tuple(boxes[i].tolist())

            # Boxes come in with normalized coordinates.  Convert to pixel values.
            # box = [box[0] * image_width, box[1] * image_height, box[2] * image_width, box[3] * image_height]

            # score = scores[i]

            # if classes[i] in model.category_index.keys():
            #     label = model.category_index[classes[i]]['name']
            # else:
            #     label = 'N/A'

            # label = classes[i]

            num_objects += 1

            print('Found object with label: "' + label + '" and score: ' + str(prob))


            # p1 = (y_min, x_min)
            # p2 = (y)

            # point1 = np.array([box[1], box[0]])
            # point2 = np.array([box[3], box[0]])
            # point3 = np.array([box[3], box[2]])
            # point4 = np.array([box[1], box[2]])

            point1 = np.array([box[1], box[0]])
            point2 = np.array([box[3], box[0]])
            point3 = np.array([box[3], box[2]])
            point4 = np.array([box[1], box[2]])

            # Add data to the output proto.
            out_obj = out_proto.object_in_image.add()
            out_obj.name = "obj" + str(num_objects) + "_label_" + label

            vertex1 = out_obj.image_properties.coordinates.vertexes.add()
            vertex1.x = point1[0]
            vertex1.y = point1[1]

            vertex2 = out_obj.image_properties.coordinates.vertexes.add()
            vertex2.x = point2[0]
            vertex2.y = point2[1]

            vertex3 = out_obj.image_properties.coordinates.vertexes.add()
            vertex3.x = point3[0]
            vertex3.y = point3[1]

            vertex4 = out_obj.image_properties.coordinates.vertexes.add()
            vertex4.x = point4[0]
            vertex4.y = point4[1]

            # Pack the confidence value.
            confidence = wrappers_pb2.FloatValue(value=prob)
            out_obj.additional_properties.Pack(confidence)

            if not args.no_debug:
                polygon = np.array([point1, point2, point3, point4], np.int32)
                polygon = polygon.reshape((-1, 1, 2))
                cv2.polylines(image, [polygon], True, (0, 255, 0), 2)

                caption = "{}: {:.3f}".format(label, prob)
                left_x = min(point1[0], min(point2[0], min(point3[0], point4[0])))
                top_y = min(point1[1], min(point2[1], min(point3[1], point4[1])))
                cv2.putText(image, caption, (int(left_x), int(top_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

        print('Found ' + str(num_objects) + ' object(s)')

        if not args.no_debug:
            debug_image_filename = './network_compute_server_output.jpg'
            cv2.imwrite(debug_image_filename, image)
            print('Wrote debug image output to: "' + debug_image_filename + '"')

        response_queue.put(out_proto)
        time.sleep(1)


class NetworkComputeBridgeWorkerServicer(
        network_compute_bridge_service_pb2_grpc.NetworkComputeBridgeWorkerServicer):

    def __init__(self, thread_input_queue, thread_output_queue):
        super(NetworkComputeBridgeWorkerServicer, self).__init__()

        self.thread_input_queue = thread_input_queue
        self.thread_output_queue = thread_output_queue

    def NetworkCompute(self, request, context):
        print('Got NetworkCompute request')
        self.thread_input_queue.put(request)
        out_proto = self.thread_output_queue.get()
        return out_proto

    def ListAvailableModels(self, request, context):
        print('Got ListAvailableModels request')
        self.thread_input_queue.put(request)
        out_proto = self.thread_output_queue.get()
        return out_proto


def register_with_robot(options):
    """ Registers this worker with the robot's Directory."""
    ip = bosdyn.client.common.get_self_ip(options.hostname)
    print('Detected IP address as: ' + ip)

    sdk = bosdyn.client.create_standard_sdk("tensorflow_server")

    robot = sdk.create_robot(options.hostname)

    # Authenticate robot before being able to use it
    bosdyn.client.util.authenticate(robot)

    directory_client = robot.ensure_client(
        bosdyn.client.directory.DirectoryClient.default_service_name)
    directory_registration_client = robot.ensure_client(
        bosdyn.client.directory_registration.DirectoryRegistrationClient.default_service_name)

    # Check to see if a service is already registered with our name
    services = directory_client.list()
    for s in services:
        if s.name == options.name:
            print("WARNING: existing service with name, \"" + options.name + "\", removing it.")
            directory_registration_client.unregister(options.name)
            break

    # Register service
    print('Attempting to register ' + ip + ':' + options.port + ' onto ' + options.hostname + ' directory...')
    directory_registration_client.register(options.name, "bosdyn.api.NetworkComputeBridgeWorker", kServiceAuthority, ip, int(options.port))



def main(argv):
    default_port = '50051'

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', help='Server\'s port number, default: ' + default_port,
                        default=default_port)
    parser.add_argument('-d', '--no-debug', help='Disable writing debug images.', action='store_true')
    parser.add_argument('-n', '--name', help='Service name', default='fetch-server')
    bosdyn.client.util.add_base_arguments(parser)

    options = parser.parse_args(argv)
    # Perform registration.
    register_with_robot(options)

    # Thread-safe queues for communication between the GRPC endpoint and the ML thread.
    request_queue = queue.Queue()
    response_queue = queue.Queue()

    # Start server thread
    thread = threading.Thread(target=process_thread, args=([options, request_queue, response_queue]))
    thread.start()

    # Set up GRPC endpoint
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    network_compute_bridge_service_pb2_grpc.add_NetworkComputeBridgeWorkerServicer_to_server(
        NetworkComputeBridgeWorkerServicer(request_queue, response_queue), server)
    server.add_insecure_port('[::]:' + options.port)
    server.start()

    print('Running...')
    thread.join()

    return True

if __name__ == '__main__':
    logging.basicConfig()
    if not main(sys.argv[1:]):
        sys.exit(1)
