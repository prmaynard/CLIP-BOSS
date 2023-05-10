# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
import sys
import time
import numpy as np
import cv2
import math
import bosdyn.client
import bosdyn.client.util
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, block_until_arm_arrives
from bosdyn.api import geometry_pb2
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.api import image_pb2
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.api import network_compute_bridge_pb2
from google.protobuf import wrappers_pb2
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import geometry_pb2, image_pb2, trajectory_pb2, world_object_pb2
from bosdyn.api import manipulation_api_pb2
from bosdyn.api import basic_command_pb2
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b,
                                         get_vision_tform_body)
from bosdyn.client.world_object import WorldObjectClient

world_object_pb2.ListWorldObjectRequest()
kImageSources = [
    'hand_color_image'
]

def get_obj_and_img(network_compute_client, server, model, confidence,
                    image_sources, label):

    for source in image_sources:
        # Build a network compute request for this image source.
        image_source_and_service = network_compute_bridge_pb2.ImageSourceAndService(
            image_source=source)

        # Input data:
        #   model name
        #   minimum confidence (between 0 and 1)
        #   if we should automatically rotate the image
        input_data = network_compute_bridge_pb2.NetworkComputeInputData(
            image_source_and_service=image_source_and_service,
            model_name=model,
            min_confidence=confidence,
            rotate_image=network_compute_bridge_pb2.NetworkComputeInputData.
            ROTATE_IMAGE_ALIGN_HORIZONTAL)

        # Server data: the service name
        server_data = network_compute_bridge_pb2.NetworkComputeServerConfiguration(
            service_name=server)

        # Pack and send the request.
        process_img_req = network_compute_bridge_pb2.NetworkComputeRequest(
            input_data=input_data, server_config=server_data)

        resp = network_compute_client.network_compute_bridge_command(
            process_img_req)

        best_obj = None
        highest_conf = 0.0
        best_vision_tform_obj = None

        img = get_bounding_box_image(resp)
        image_full = resp.image_response

        # Show the image
        cv2.imshow("Fetch", img)
        cv2.waitKey(15)

        if len(resp.object_in_image) > 0:
            for obj in resp.object_in_image:
                # Get the label
                obj_label = obj.name.split('_label_')[-1]
                if obj_label != label:
                    continue
                conf_msg = wrappers_pb2.FloatValue()
                obj.additional_properties.Unpack(conf_msg)
                conf = conf_msg.value

                try:
                    vision_tform_obj = frame_helpers.get_a_tform_b(
                        obj.transforms_snapshot,
                        frame_helpers.VISION_FRAME_NAME,
                        obj.image_properties.frame_name_image_coordinates)
                except bosdyn.client.frame_helpers.ValidateFrameTreeError:
                    # No depth data available.
                    vision_tform_obj = None

                if conf > highest_conf and vision_tform_obj is not None:
                    highest_conf = conf
                    best_obj = obj
                    best_vision_tform_obj = vision_tform_obj

        if best_obj is not None:
            return best_obj, image_full, best_vision_tform_obj

    return None, None, None

def get_bounding_box_image(response):
    dtype = np.uint8
    img = np.frombuffer(response.image_response.shot.image.data, dtype=dtype)
    if response.image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(response.image_response.shot.image.rows,
                          response.image_response.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)

    # Convert to BGR so we can draw colors
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw bounding boxes in the image for all the detections.
    for obj in response.object_in_image:
        conf_msg = wrappers_pb2.FloatValue()
        obj.additional_properties.Unpack(conf_msg)
        confidence = conf_msg.value

        polygon = []
        min_x = float('inf')
        min_y = float('inf')
        for v in obj.image_properties.coordinates.vertexes:
            polygon.append([v.x, v.y])
            min_x = min(min_x, v.x)
            min_y = min(min_y, v.y)

        polygon = np.array(polygon, np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

        caption = "{} {:.3f}".format(obj.name, confidence)
        cv2.putText(img, caption, (int(min_x), int(min_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

def find_center_px(polygon):
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for vert in polygon.vertexes:
        if vert.x < min_x:
            min_x = vert.x
        if vert.y < min_y:
            min_y = vert.y
        if vert.x > max_x:
            max_x = vert.x
        if vert.y > max_y:
            max_y = vert.y
    x = math.fabs(max_x - min_x) / 2.0 + min_x
    y = math.fabs(max_y - min_y) / 2.0 + min_y
    return (x, y)

def block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=None, verbose=False):
    """Helper that blocks until a trajectory command reaches STATUS_AT_GOAL or a timeout is
        exceeded.
       Args:
        command_client: robot command client, used to request feedback
        cmd_id: command ID returned by the robot when the trajectory command was sent
        timeout_sec: optional number of seconds after which we'll return no matter what the
                        robot's state is.
        verbose: if we should print state at 10 Hz.
       Return values:
        True if reaches STATUS_AT_GOAL, False otherwise.
    """
    start_time = time.time()

    if timeout_sec is not None:
        end_time = start_time + timeout_sec
        now = time.time()

    while timeout_sec is None or now < end_time:
        feedback_resp = command_client.robot_command_feedback(cmd_id)

        current_state = feedback_resp.feedback.mobility_feedback.se2_trajectory_feedback.status

        if verbose:
            current_state_str = basic_command_pb2.SE2TrajectoryCommand.Feedback.Status.Name(current_state)

            current_time = time.time()
            print('Walking: ({time:.1f} sec): {state}'.format(
                time=current_time - start_time, state=current_state_str),
                  end='                \r')

        if current_state == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL:
            return True

        time.sleep(0.1)
        now = time.time()

    if verbose:
        print('block_for_trajectory_cmd: timeout exceeded.')

    return False

def main(argv):
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        '-s',
        '--ml-service',
        help='Service name of external machine learning server.',
        required=True)
    parser.add_argument('-m',
                        '--model',
                        help='Model name running on the external server.',
                        required=True)
    parser.add_argument(
        '-p',
        '--person-model',
        help='Person detection model name running on the external server.')
    parser.add_argument('-c',
                        '--confidence-dogtoy',
                        help='Minimum confidence to return an object for the dogoy (0.0 to 1.0)',
                        default=0.5,
                        type=float)
    parser.add_argument('-e',
                        '--confidence-person',
                        help='Minimum confidence for person detection (0.0 to 1.0)',
                        default=0.6,
                        type=float)
    options = parser.parse_args(argv)

    cv2.namedWindow("Fetch")
    cv2.waitKey(500)

    sdk = bosdyn.client.create_standard_sdk('SpotFetchClient')
    sdk.register_service_client(NetworkComputeBridgeClient)
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    # Time sync is necessary so that time-based filter requests can be converted
    robot.time_sync.wait_for_sync()

    network_compute_client = robot.ensure_client(
        NetworkComputeBridgeClient.default_service_name)
    robot_state_client = robot.ensure_client(
        RobotStateClient.default_service_name)
    command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    manipulation_api_client = robot.ensure_client(
        ManipulationApiClient.default_service_name)
    
    robot_id = robot.ensure_client(RobotIdClient.default_service_name).get_id(timeout=0.4)
    power_client = robot.ensure_client(PowerClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)
    fiducialFollower = FollowFiducial(robot, robot_id, power_client, image_client, robot_state_client, command_client, world_object_client)
    # This script assumes the robot is already standing via the tablet.  We'll take over from the
    # tablet.
    lease_client.take()
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Store the position of the hand at the last toy drop point.
        vision_tform_hand_at_drop = None

        while True:
            holding_toy = False
            while not holding_toy:
                # Capture an image and run ML on it.
                dogtoy, image, vision_tform_dogtoy = get_obj_and_img(
                    network_compute_client, options.ml_service, options.model,
                    options.confidence_dogtoy, kImageSources, 'striped ball')

                if dogtoy is None:
                    # Didn't find anything, keep searching.
                    continue

                # If we have already dropped the toy off, make sure it has moved a sufficient amount before
                # picking it up again
                if vision_tform_hand_at_drop is not None and pose_dist(
                        vision_tform_hand_at_drop, vision_tform_dogtoy) < 0.5:
                    print('Found dogtoy, but it hasn\'t moved.  Waiting...')
                    time.sleep(1)
                    continue

                print('Found dogtoy...')

                # Got a dogtoy.  Request pick up.

                # Stow the arm in case it is deployed
                stow_cmd = RobotCommandBuilder.arm_stow_command()
                command_client.robot_command(stow_cmd)

                # NOTE: we'll enable this code in Part 5, when we understand it.
                # -------------------------
                # # Walk to the object.
                # walk_rt_vision, heading_rt_vision = compute_stand_location_and_yaw(
                    # vision_tform_dogtoy, robot_state_client, distance_margin=1.0)

                # move_cmd = RobotCommandBuilder.trajectory_command(
                    # goal_x=walk_rt_vision[0],
                    # goal_y=walk_rt_vision[1],
                    # goal_heading=heading_rt_vision,
                    # frame_name=frame_helpers.VISION_FRAME_NAME,
                    # params=get_walking_params(0.5, 0.5))
                # end_time = 5.0
                # cmd_id = command_client.robot_command(command=move_cmd,
                                                      # end_time_secs=time.time() +
                                                      # end_time)

                # # Wait until the robot reports that it is at the goal.
                # block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=5, verbose=True)
                # -------------------------
RobotCommandBuilder.trajectory_command()
                # The ML result is a bounding box.  Find the center.
                (center_px_x,
                 center_px_y) = find_center_px(dogtoy.image_properties.coordinates)

                # Request Pick Up on that pixel.
                pick_vec = geometry_pb2.Vec2(x=center_px_x, y=center_px_y)
                grasp = manipulation_api_pb2.PickObjectInImage(
                    pixel_xy=pick_vec,
                    transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                    frame_name_image_sensor=image.shot.frame_name_image_sensor,
                    camera_model=image.source.pinhole)

                # We can specify where in the gripper we want to grasp. About halfway is generally good for
                # small objects like this. For a bigger object like a shoe, 0 is better (use the entire
                # gripper)
                grasp.grasp_params.grasp_palm_to_fingertip = 0.6

                # Tell the grasping system that we want a top-down grasp.

                # Add a constraint that requests that the x-axis of the gripper is pointing in the
                # negative-z direction in the vision frame.

                # The axis on the gripper is the x-axis.
                axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

                # The axis in the vision frame is the negative z-axis
                axis_to_align_with_ewrt_vision = geometry_pb2.Vec3(x=0, y=0, z=-1)

                # Add the vector constraint to our proto.
                constraint = grasp.grasp_params.allowable_orientation.add()
                constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
                    axis_on_gripper_ewrt_gripper)
                constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
                    axis_to_align_with_ewrt_vision)

                # We'll take anything within about 15 degrees for top-down or horizontal grasps.
                constraint.vector_alignment_with_tolerance.threshold_radians = 0.25

                # Specify the frame we're using.
                grasp.grasp_params.grasp_params_frame_name = frame_helpers.VISION_FRAME_NAME

                # Build the proto
                grasp_request = manipulation_api_pb2.ManipulationApiRequest(
                    pick_object_in_image=grasp)

                # Send the request
                print('Sending grasp request...')
                cmd_response = manipulation_api_client.manipulation_api_command(
                    manipulation_api_request=grasp_request)

                # Wait for the grasp to finish
                grasp_done = False
                failed = False
                time_start = time.time()
                while not grasp_done:
                    feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                        manipulation_cmd_id=cmd_response.manipulation_cmd_id)

                    # Send a request for feedback
                    response = manipulation_api_client.manipulation_api_feedback_command(
                        manipulation_api_feedback_request=feedback_request)

                    current_state = response.current_state
                    current_time = time.time() - time_start
                    print('Current state ({time:.1f} sec): {state}'.format(
                        time=current_time,
                        state=manipulation_api_pb2.ManipulationFeedbackState.Name(
                            current_state)),
                          end='                \r')
                    sys.stdout.flush()

                    failed_states = [manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                                     manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
                                     manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP,
                                     manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_WAITING_DATA_AT_EDGE]

                    failed = current_state in failed_states
                    grasp_done = current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or failed

                    time.sleep(0.1)

                holding_toy = not failed

            # Move the arm to a carry position.
            print('')
            print('Grasp finished, search for a person...')
            carry_cmd = RobotCommandBuilder.arm_carry_command()
            command_client.robot_command(carry_cmd)

            # Wait for the carry command to finish
            time.sleep(0.75)

            # For now, we'll just exit...
            print('')
            print('Done for now, returning control to tablet in 5 seconds...')
            time.sleep(1.0)

            # fiducialFollower.go_to_fudicial('99')
            
            
import logging
import math
import signal
import sys
import threading
import time
from sys import platform

import cv2
import numpy as np
from PIL import Image

import bosdyn.client
import bosdyn.client.util
from bosdyn import geometry
from bosdyn.api import geometry_pb2, image_pb2, trajectory_pb2, world_object_pb2
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b,
                                         get_vision_tform_body)
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_id import RobotIdClient, version_tuple
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient
BODY_LENGTH = 1.1
class FollowFiducial(object):
    """ Detect and follow a fiducial with Spot."""

    def __init__(self, robot, robot_id, power_client, image_client, state_client, command_client, world_object):
        # Robot instance variable.
        self._robot = robot
        self._robot_id = robot_id
        self._power_client = power_client
        self._image_client = image_client
        self._robot_state_client = state_client
        self._robot_command_client = command_client
        self._world_object_client = world_object

        # Stopping Distance (x,y) offset from the tag and angle offset from desired angle.
        self._tag_offset = float(0.1) + BODY_LENGTH / 2.0  # meters

        # Maximum speeds.
        self._max_x_vel = 0.5
        self._max_y_vel = 0.5
        self._max_ang_vel = 1.0

        # Indicator if fiducial detection's should be from the world object service using
        # spot's perception system or detected with the apriltag library. If the software version
        # does not include the world object service, than default to april tag library.
        self._use_world_object_service = (True and
                                          self.check_if_version_has_world_objects(self._robot_id))

        # Indicators for movement and image displays.
        self._standup = True  # Stand up the robot.
        self._movement_on = True  # Let the robot walk towards the fiducial.
        self._limit_speed = True  # Limit the robot's walking speed.
        self._avoid_obstacles = False  # Disable obstacle avoidance.

        # Epsilon distance between robot and desired go-to point.
        self._x_eps = .05
        self._y_eps = .05
        self._angle_eps = .075

        # Indicator for if motor power is on.
        self._powered_on = False

        # Counter for the number of iterations completed.
        self._attempts = 0

        # Maximum amount of iterations before powering off the motors.
        self._max_attempts = 100000

        # Camera intrinsics for the current camera source being analyzed.
        self._intrinsics = None

        # Transform from the robot's camera frame to the baselink frame.
        # It is a math_helpers.SE3Pose.
        self._camera_tform_body = None

        # Transform from the robot's baselink to the world frame.
        # It is a math_helpers.SE3Pose.
        self._body_tform_world = None

        # Latest detected fiducial's position in the world.
        self._current_tag_world_pose = np.array([])

        # Heading angle based on the camera source which detected the fiducial.
        self._angle_desired = None

        # Dictionary mapping camera source to it's latest image taken.
        self._image = dict()

        # List of all possible camera sources.
        self._source_names = [
            src.name for src in self._image_client.list_image_sources() if
            (src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL and "depth" not in src.name)
        ]
        print(self._source_names)

        # Dictionary mapping camera source to previously computed extrinsics.
        self._camera_to_extrinsics_guess = self.populate_source_dict()

        # Camera source which a bounding box was last detected in.
        self._previous_source = None

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_client.get_robot_state()

    @property
    def image(self):
        """Return the current image associated with each source name."""
        return self._image

    @property
    def image_sources_list(self):
        """Return the list of camera sources."""
        return self._source_names

    def populate_source_dict(self):
        """Fills dictionary of the most recently computed camera extrinsics with the camera source.
           The initial boolean indicates if the extrinsics guess should be used."""
        camera_to_extrinsics_guess = dict()
        for src in self._source_names:
            # Dictionary values: use_extrinsics_guess bool, (rotation vector, translation vector) tuple.
            camera_to_extrinsics_guess[src] = (False, (None, None))
        return camera_to_extrinsics_guess

    def check_if_version_has_world_objects(self, robot_id):
        """Check that software version contains world object service."""
        # World object service was released in spot-sdk version 1.2.0
        return version_tuple(robot_id.software_release.version) >= (1, 2, 0)

    def go_to_fudicial(self, number):
        """Claim lease of robot and start the fiducial follower."""
        self._robot.time_sync.wait_for_sync()

        # Stand the robot up.
        if self._standup:
            self.power_on()
            blocking_stand(self._robot_command_client)

            # Delay grabbing image until spot is standing (or close enough to upright).
            time.sleep(.35)

        while self._attempts <= self._max_attempts:
            detected_fiducial = False
            fiducial_rt_world = None
            if self._use_world_object_service:
                # Get the first fiducial object Spot detects with the world object service.
                fiducial = self.get_fiducial_objects(number)
                if fiducial is not None:
                    vision_tform_fiducial = get_a_tform_b(
                        fiducial.transforms_snapshot, VISION_FRAME_NAME,
                        fiducial.apriltag_properties.frame_name_fiducial).to_proto()
                    if vision_tform_fiducial is not None:
                        detected_fiducial = True
                        fiducial_rt_world = vision_tform_fiducial.position
                    detected_fiducial = True

            if detected_fiducial:
                # Go to the tag and stop within a certain distance
                self.go_to_tag(fiducial_rt_world)
            else:
                print("No fiducials found")

            self._attempts += 1  #increment attempts at finding a fiducial

        # Power off at the conclusion of the example.

    def get_fiducial_objects(self, number):
        """Get all fiducials that Spot detects with its perception system."""
        # Get all fiducial objects (an object of a specific type).
        request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        fiducial_objects = self._world_object_client.list_world_objects(
            object_type=request_fiducials).world_objects
        for fiducial in fiducial_objects:
            if fiducial.name.split('_')[-1] == number:
                return fiducial
        # Return none if no fiducials are found.
        return None

    def image_to_bounding_box(self):
        """Determine which camera source has a fiducial.
           Return the bounding box of the first detected fiducial."""
        #Iterate through all five camera sources to check for a fiducial
        for i in range(len(self._source_names) + 1):
            # Get the image from the source camera.
            if i == 0:
                if self._previous_source != None:
                    # Prioritize the camera the fiducial was last detected in.
                    source_name = self._previous_source
                else:
                    continue
            elif self._source_names[i - 1] == self._previous_source:
                continue
            else:
                source_name = self._source_names[i - 1]

            img_req = build_image_request(source_name, quality_percent=100,
                                          image_format=image_pb2.Image.FORMAT_RAW)
            image_response = self._image_client.get_image([img_req])
            self._camera_tform_body = get_a_tform_b(image_response[0].shot.transforms_snapshot,
                                                    image_response[0].shot.frame_name_image_sensor,
                                                    BODY_FRAME_NAME)
            self._body_tform_world = get_a_tform_b(image_response[0].shot.transforms_snapshot,
                                                   BODY_FRAME_NAME, VISION_FRAME_NAME)

            # Camera intrinsics for the given source camera.
            self._intrinsics = image_response[0].source.pinhole.intrinsics
            width = image_response[0].shot.image.cols
            height = image_response[0].shot.image.rows

            # detect given fiducial in image and return the bounding box of it
            bboxes = self.detect_fiducial_in_image(image_response[0].shot.image, (width, height),
                                                   source_name)
            if bboxes:
                print("Found bounding box for " + str(source_name))
                return bboxes, source_name
            else:
                self._tag_not_located = True
                print("Failed to find bounding box for " + str(source_name))
        return [], None

    def detect_fiducial_in_image(self, image, dim, source_name):
        """Detect the fiducial within a single image and return its bounding box."""
        image_grey = np.array(
            Image.frombytes('P', (int(dim[0]), int(dim[1])), data=image.data, decoder_name='raw'))

        #Rotate each image such that it is upright
        image_grey = self.rotate_image(image_grey, source_name)

        #Make the image greyscale to use bounding box detections
        detector = apriltag(family="tag36h11")
        detections = detector.detect(image_grey)

        bboxes = []
        for i in range(len(detections)):
            # Draw the bounding box detection in the image.
            bbox = detections[i]['lb-rb-rt-lt']
            cv2.polylines(image_grey, [np.int32(bbox)], True, (0, 0, 0), 2)
            bboxes.append(bbox)

        self._image[source_name] = image_grey
        return bboxes

    def bbox_to_image_object_pts(self, bbox):
        """Determine the object points and image points for the bounding box.
           The origin in object coordinates = top left corner of the fiducial.
           Order both points sets following: (TL,TR, BL, BR)"""
        fiducial_height_and_width = 146  #mm
        obj_pts = np.array([[0, 0], [fiducial_height_and_width, 0], [0, fiducial_height_and_width],
                            [fiducial_height_and_width, fiducial_height_and_width]],
                           dtype=np.float32)
        #insert a 0 as the third coordinate (xyz)
        obj_points = np.insert(obj_pts, 2, 0, axis=1)

        #['lb-rb-rt-lt']
        img_pts = np.array([[bbox[3][0], bbox[3][1]], [bbox[2][0], bbox[2][1]],
                            [bbox[0][0], bbox[0][1]], [bbox[1][0], bbox[1][1]]], dtype=np.float32)
        return obj_points, img_pts

    def pixel_coords_to_camera_coords(self, bbox, intrinsics, source_name):
        """Compute transformation of 2d pixel coordinates to 3d camera coordinates."""
        camera = self.make_camera_matrix(intrinsics)
        # Track a triplet of (translation vector, rotation vector, camera source name)
        best_bbox = (None, None, source_name)
        # The best bounding box is considered the closest to the robot body.
        closest_dist = float('inf')
        for i in range(len(bbox)):
            obj_points, img_points = self.bbox_to_image_object_pts(bbox[i])
            if self._camera_to_extrinsics_guess[source_name][0]:
                # initialize the position estimate with the previous extrinsics solution
                # then iteratively solve for new position
                old_rvec, old_tvec = self._camera_to_extrinsics_guess[source_name][1]
                _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)),
                                             old_rvec, old_tvec, True, cv2.SOLVEPNP_ITERATIVE)
            else:
                # Determine current extrinsic solution for the tag.
                _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)))

            # Save extrinsics results to help speed up next attempts to locate bounding box in
            # the same camera source.
            self._camera_to_extrinsics_guess[source_name] = (True, (rvec, tvec))

            dist = math.sqrt(float(tvec[0][0])**2 + float(tvec[1][0])**2 +
                             float(tvec[2][0])**2) / 1000.0
            if dist < closest_dist:
                closest_dist = dist
                best_bbox = (tvec, rvec, source_name)

        # Flag indicating if the best april tag been found/located
        self._tag_not_located = best_bbox[0] is None and best_bbox[1] is None
        return best_bbox

    def compute_fiducial_in_world_frame(self, tvec):
        """Transform the tag position from camera coordinates to world coordinates."""
        fiducial_rt_camera_frame = np.array(
            [float(tvec[0][0]) / 1000.0,
             float(tvec[1][0]) / 1000.0,
             float(tvec[2][0]) / 1000.0])
        body_tform_fiducial = (self._camera_tform_body.inverse()).transform_point(
            fiducial_rt_camera_frame[0], fiducial_rt_camera_frame[1], fiducial_rt_camera_frame[2])
        fiducial_rt_world = self._body_tform_world.inverse().transform_point(
            body_tform_fiducial[0], body_tform_fiducial[1], body_tform_fiducial[2])
        return fiducial_rt_world

    def go_to_tag(self, fiducial_rt_world):
        """Use the position of the april tag in vision world frame and command the robot."""
        # Compute the go-to point (offset by .5m from the fiducial position) and the heading at
        # this point.
        self._current_tag_world_pose, self._angle_desired = self.offset_tag_pose(
            fiducial_rt_world, self._tag_offset)

        #Command the robot to go to the tag in kinematic odometry frame
        mobility_params = self.set_mobility_params()
        tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=self._current_tag_world_pose[0], goal_y=self._current_tag_world_pose[1],
            goal_heading=self._angle_desired, frame_name=VISION_FRAME_NAME, params=mobility_params,
            body_height=0.0, locomotion_hint=spot_command_pb2.HINT_AUTO)
        end_time = 5.0
        if self._movement_on and self._powered_on:
            #Issue the command to the robot
            self._robot_command_client.robot_command(lease=None, command=tag_cmd,
                                                     end_time_secs=time.time() + end_time)
            # #Feedback to check and wait until the robot is in the desired position or timeout
            start_time = time.time()
            current_time = time.time()
            while (not self.final_state() and current_time - start_time < end_time):
                time.sleep(.25)
                current_time = time.time()
        return

    def final_state(self):
        """Check if the current robot state is within range of the fiducial position."""
        robot_state = get_vision_tform_body(self.robot_state.kinematic_state.transforms_snapshot)
        robot_angle = robot_state.rot.to_yaw()
        if self._current_tag_world_pose.size != 0:
            x_dist = abs(self._current_tag_world_pose[0] - robot_state.x)
            y_dist = abs(self._current_tag_world_pose[1] - robot_state.y)
            angle = abs(self._angle_desired - robot_angle)
            if ((x_dist < self._x_eps) and (y_dist < self._y_eps) and (angle < self._angle_eps)):
                return True
        return False

    def get_desired_angle(self, xhat):
        """Compute heading based on the vector from robot to object."""
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, xhat)
        mat = np.array([xhat, yhat, zhat]).transpose()
        return Quat.from_matrix(mat).to_yaw()

    def offset_tag_pose(self, object_rt_world, dist_margin=1.0):
        """Offset the go-to location of the fiducial and compute the desired heading."""
        robot_rt_world = get_vision_tform_body(self.robot_state.kinematic_state.transforms_snapshot)
        robot_to_object_ewrt_world = np.array(
            [object_rt_world.x - robot_rt_world.x, object_rt_world.y - robot_rt_world.y, 0])
        robot_to_object_ewrt_world_norm = robot_to_object_ewrt_world / np.linalg.norm(
            robot_to_object_ewrt_world)
        heading = self.get_desired_angle(robot_to_object_ewrt_world_norm)
        goto_rt_world = np.array([
            object_rt_world.x - robot_to_object_ewrt_world_norm[0] * dist_margin,
            object_rt_world.y - robot_to_object_ewrt_world_norm[1] * dist_margin
        ])
        return goto_rt_world, heading

    def set_mobility_params(self):
        """Set robot mobility params to disable obstacle avoidance."""
        obstacles = spot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=True,
                                                    disable_vision_foot_obstacle_avoidance=True,
                                                    disable_vision_foot_constraint_avoidance=True,
                                                    obstacle_avoidance_padding=.001)
        body_control = self.set_default_body_control()
        if self._limit_speed:
            speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
                linear=Vec2(x=self._max_x_vel, y=self._max_y_vel), angular=self._max_ang_vel))
            if not self._avoid_obstacles:
                mobility_params = spot_command_pb2.MobilityParams(
                    obstacle_params=obstacles, vel_limit=speed_limit, body_control=body_control,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
            else:
                mobility_params = spot_command_pb2.MobilityParams(
                    vel_limit=speed_limit, body_control=body_control,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
        elif not self._avoid_obstacles:
            mobility_params = spot_command_pb2.MobilityParams(
                obstacle_params=obstacles, body_control=body_control,
                locomotion_hint=spot_command_pb2.HINT_AUTO)
        else:
            #When set to none, RobotCommandBuilder populates with good default values
            mobility_params = None
        return mobility_params

    @staticmethod
    def set_default_body_control():
        """Set default body control params to current body position"""
        footprint_R_body = geometry.EulerZXY()
        position = geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0)
        rotation = footprint_R_body.to_quaternion()
        pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)
        point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
        traj = trajectory_pb2.SE3Trajectory(points=[point])
        return spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)

    @staticmethod
    def rotate_image(image, source_name):
        """Rotate the image so that it is always displayed upright."""
        if source_name == "frontleft_fisheye_image":
            image = cv2.rotate(image, rotateCode=0)
        elif source_name == "right_fisheye_image":
            image = cv2.rotate(image, rotateCode=1)
        elif source_name == "frontright_fisheye_image":
            image = cv2.rotate(image, rotateCode=0)
        return image

    @staticmethod
    def make_camera_matrix(ints):
        """Transform the ImageResponse proto intrinsics into a camera matrix."""
        camera_matrix = np.array([[ints.focal_length.x, ints.skew.x, ints.principal_point.x],
                                  [ints.skew.y, ints.focal_length.y, ints.principal_point.y],
                                  [0, 0, 1]])
        return camera_matrix


class DisplayImagesAsync(object):
    """Display the images Spot sees from all five cameras."""

    def __init__(self, fiducial_follower):
        self._fiducial_follower = fiducial_follower
        self._thread = None
        self._started = False
        self._sources = []

    def get_image(self):
        """Retrieve current images (with bounding boxes) from the fiducial detector."""
        images = self._fiducial_follower.image
        image_by_source = []
        for s_name in self._sources:
            if s_name in images:
                image_by_source.append(images[s_name])
            else:
                image_by_source.append(np.array([]))
        return image_by_source

    def start(self):
        """Initialize the thread to display the images."""
        if self._started:
            return None
        self._sources = self._fiducial_follower.image_sources_list
        self._started = True
        self._thread = threading.Thread(target=self.update)
        self._thread.start()
        return self

    def update(self):
        """Update the images being displayed to match that seen by the robot."""
        while self._started:
            images = self.get_image()
            for i, image in enumerate(images):
                if image.size != 0:
                    original_height, original_width = image.shape[:2]
                    resized_image = cv2.resize(
                        image, (int(original_width * .5), int(original_height * .5)),
                        interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(self._sources[i], resized_image)
                    cv2.moveWindow(self._sources[i],
                                   max(int(i * original_width * .5), int(i * original_height * .5)),
                                   0)
                    cv2.waitKey(1)

    def stop(self):
        """Stop the thread and the image displays."""
        self._started = False
        cv2.destroyAllWindows()


class Exit(object):
    """Handle exiting on SIGTERM."""

    def __init__(self):
        self._kill_now = False
        signal.signal(signal.SIGTERM, self._sigterm_handler)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False

    def _sigterm_handler(self, _signum, _frame):
        self._kill_now = True

    @property
    def kill_now(self):
        """Return if sigterm received and program should end."""
        return self._kill_now


# def main():
#     """Command-line interface."""
#     import argparse

#     parser = argparse.ArgumentParser()
#     bosdyn.client.util.add_base_arguments(parser)
#     parser.add_argument("--distance-margin", default=.5,
#                         help="Distance [meters] that the robot should stop from the fiducial.")
#     parser.add_argument("--limit-speed", default=True, type=lambda x: (str(x).lower() == 'true'),
#                         help="If the robot should limit its maximum speed.")
#     parser.add_argument("--avoid-obstacles", default=False, type=lambda x:
#                         (str(x).lower() == 'true'),
#                         help="If the robot should have obstacle avoidance enabled.")
#     parser.add_argument(
#         "--use-world-objects", default=True, type=lambda x: (str(x).lower() == 'true'),
#         help="If fiducials should be from the world object service or the apriltag library.")
#     options = parser.parse_args()

#     # If requested, attempt import of Apriltag library
#     if not options.use_world_objects:
#         try:
#             global apriltag
#             from apriltag import apriltag
#         except ImportError as e:
#             print("Could not import the AprilTag library. Aborting. Exception: ", str(e))
#             return False

#     # Create robot object.
#     sdk = create_standard_sdk('FollowFiducialClient')
#     robot = sdk.create_robot(options.hostname)

#     fiducial_follower = None
#     image_viewer = None
#     try:
#         with Exit():
#             bosdyn.client.util.authenticate(robot)
#             robot.start_time_sync()

#             # Verify the robot is not estopped.
#             assert not robot.is_estopped(), "Robot is estopped. " \
#                                             "Please use an external E-Stop client, " \
#                                             "such as the estop SDK example, to configure E-Stop."

#             fiducial_follower = FollowFiducial(robot, options)
#             time.sleep(.1)
#             if not options.use_world_objects and str.lower(sys.platform) != "darwin":
#                 # Display the detected bounding boxes on the images when using the april tag library.
#                 # This is disabled for MacOS-X operating systems.
#                 image_viewer = DisplayImagesAsync(fiducial_follower)
#                 image_viewer.start()
#             lease_client = robot.ensure_client(LeaseClient.default_service_name)
#             with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True,
#                                                     return_at_exit=True):
#                 fiducial_follower.start()
#     except RpcError as err:
#         LOGGER.error("Failed to communicate with robot: %s", err)
#     finally:
#         if image_viewer is not None:
#             image_viewer.stop()

#     return False

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
