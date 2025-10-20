#!/usr/bin/env python3

import math
import random
from enum import Enum

import cv2  # OpenCV2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

################# PLANNING ##########################
import numpy as np
from collections import deque

################# PERCEPTION #########################
from cave_explorer.artifact_utils.save_images import save_image
import time

from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
####################################################

def wrap_angle(angle):
    """Function to wrap an angle between 0 and 2*Pi"""
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def pose2d_to_pose(pose_2d):
    """Convert a Pose2D to a full 3D Pose"""
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose

class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    # Add more!
    ####################################################
    EXPLORE_FRONTIERS = 6 # Planning 1
    ####################################################

class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')

        # Variables/Flags for mapping
        self.xlim_ = [0.0, 0.0]
        self.ylim_ = [0.0, 0.0]

        # Variables/Flags for perception
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False

        # Marker for artifact locations
        # See https://wiki.ros.org/rviz/DisplayTypes/Marker
        self.marker_artifacts_ = Marker()
        self.marker_artifacts_.header.frame_id = "map"
        self.marker_artifacts_.ns = "artifacts"
        self.marker_artifacts_.id = 0
        self.marker_artifacts_.type = Marker.SPHERE_LIST
        self.marker_artifacts_.action = Marker.ADD
        self.marker_artifacts_.pose.position.x = 0.0
        self.marker_artifacts_.pose.position.y = 0.0
        self.marker_artifacts_.pose.position.z = 0.0
        self.marker_artifacts_.pose.orientation.x = 0.0
        self.marker_artifacts_.pose.orientation.y = 0.0
        self.marker_artifacts_.pose.orientation.z = 0.0
        self.marker_artifacts_.pose.orientation.w = 1.0
        self.marker_artifacts_.scale.x = 1.0
        self.marker_artifacts_.scale.y = 1.0
        self.marker_artifacts_.scale.z = 1.0
        self.marker_artifacts_.color.a = 1.0
        self.marker_artifacts_.color.r = 0.0
        self.marker_artifacts_.color.g = 1.0
        self.marker_artifacts_.color.b = 0.2
        self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)

        # Remember the artifact locations
        # Array of type geometry_msgs.Point
        self.artifact_locations_ = []

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Prepare transformation to get robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Action client for nav2
        self.nav2_action_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().warn('Waiting for navigate_to_pose action...')
        self.nav2_action_client_.wait_for_server()
        self.get_logger().warn('navigate_to_pose connected')
        self.ready_for_next_goal_ = True
        self.declare_parameter('print_feedback', rclpy.Parameter.Type.BOOL)

        # Publisher for the goal pose visualisation
        self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose', 1)

        # Subscribe to the map topic to get current bounds
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map',  self.map_callback, 1)

        # Prepare image processing
        self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)
        self.declare_parameter('computer_vision_model_filename', rclpy.Parameter.Type.STRING)
        self.computer_vision_model_ = cv2.CascadeClassifier(self.get_parameter('computer_vision_model_filename').value)
        self.image_sub_ = self.create_subscription(Image, '/camera/image', self.image_callback, 1)


        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)

        self.last_scan_time_ = self.get_clock().now()  # Track when last scan occurred

        #################################################################################################
        # Map cache (for frontier detection)
        self.map_msg_ = None
        self.map_resolution_ = None
        self.map_origin_xy_ = None   # (ox, oy)
        self.map_size_ = None        # (width, height)
        self.map_grid_ = None        # numpy int8 array shape (H,W) with {-1,0,100}

        # Frontier markers
        self.marker_frontiers_ = Marker()
        self.marker_frontiers_.header.frame_id = "map"
        self.marker_frontiers_.ns = "frontiers"
        self.marker_frontiers_.id = 0
        self.marker_frontiers_.type = Marker.POINTS
        self.marker_frontiers_.action = Marker.ADD
        self.marker_frontiers_.scale.x = 0.25
        self.marker_frontiers_.scale.y = 0.25
        self.marker_frontiers_.color.a = 1.0
        self.marker_frontiers_.color.r = 1.0
        self.marker_frontiers_.color.g = 1.0
        self.marker_frontiers_.color.b = 0.0

        self.marker_frontier_goal_ = Marker()
        self.marker_frontier_goal_.header.frame_id = "map"
        self.marker_frontier_goal_.ns = "frontier_goal"
        self.marker_frontier_goal_.id = 0
        self.marker_frontier_goal_.type = Marker.SPHERE_LIST
        self.marker_frontier_goal_.action = Marker.ADD
        self.marker_frontier_goal_.scale.x = 0.5
        self.marker_frontier_goal_.scale.y = 0.5
        self.marker_frontier_goal_.scale.z = 0.5
        self.marker_frontier_goal_.color.a = 1.0
        self.marker_frontier_goal_.color.r = 0.2
        self.marker_frontier_goal_.color.g = 0.8
        self.marker_frontier_goal_.color.b = 0.2

        # Frontier selection weights (tune to taste)
        self.w_size = 2.0     # reward for bigger frontier clusters
        self.w_dist = 1.0     # penalty per meter of distance to frontier

        # --- Coverage / revisit control ---
        self.visit_counts_ = None     # np.int32 (H, W), initialised in map_callback
        self.visit_decay_  = 0.997    # decay per tick so trails fade over time (0.99–0.999)
        self.w_visit_local = 1.5      # penalty for standing on a “busy” area (per unit avg count)
        self.w_visit_path  = 0.8      # penalty for reusing travelled corridors (integral along path)
        self.kernel_radius_cells = 2  # “splat” radius when incrementing visits

        # Perception state
        self.last_scan_time_ = self.get_clock().now()
        self.current_pose_ = None
        self.last_image_ = None
        self.last_xyxy = None
        self.latest_map_ = None   # store raw map msg for raycast in localisation

        
        # --- YOLOv8 (Simplified Ultralytics API) ---
        try:
            self.yolo_model_ = YOLO('/home/student/ros_ws/src/cave_explorer/models/artifacts/best.pt')
            self.get_logger().info('✅ Loaded YOLOv8 model from best.pt')
            self.get_logger().info(f"Model classes: {self.yolo_model_.names}")
        except Exception as e:
            self.get_logger().warn(f'⚠️ Failed to load YOLO model: {e}. Falling back to pretrained COCO model.')
            self.yolo_model_ = YOLO('yolov8n.pt')

        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model_.to(self.device)
        self.get_logger().info(f"🚀 YOLO model running on {self.device.upper()}")

        self.marker_artifacts_est_ = Marker()
        self.marker_artifacts_est_.header.frame_id = "map"
        self.marker_artifacts_est_.ns = "artifact_estimates"
        self.marker_artifacts_est_.id = 0
        self.marker_artifacts_est_.type = Marker.SPHERE_LIST
        self.marker_artifacts_est_.action = Marker.ADD
        self.marker_artifacts_est_.scale.x = 0.6
        self.marker_artifacts_est_.scale.y = 0.6
        self.marker_artifacts_est_.scale.z = 0.6
        self.marker_artifacts_est_.color.a = 1.0
        self.marker_artifacts_est_.color.r = 1.0  # yellow
        self.marker_artifacts_est_.color.g = 1.0
        self.marker_artifacts_est_.color.b = 0.0

        self.artifact_estimates_ = []  # Point[]

        # ---- Perception filters ----
        self.allowed_classes = {"stop sign"}  # only log these
        self.min_conf = 0.55                  # tune up if you still get noise
        self.edge_frac = 0.06                 # ignore boxes too close to image edges (6%)
        self.min_box_area = 22 * 22           # ignore tiny blobs
        self.max_box_frac = 0.28              # ignore absurdly large boxes (likely self/blur)
        self.hfov_deg = 70.0                  # camera horizontal FOV (adjust to your camera)
        #################################################################################################
    
    def get_pose_2d(self):
        """Get the 2d pose of the robot"""

        # Lookup the latest transform
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f'Could not transform: {ex}')
            return

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = t.transform.translation.x
        pose.y = t.transform.translation.y

        qw = t.transform.rotation.w
        qz = t.transform.rotation.z

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw))

        self.get_logger().warn(f'Pose: {pose}')

        return pose

    def map_callback(self, map_msg: OccupancyGrid):
        """Cache full map + bounds for frontier detection and goal validation."""
        self.map_msg_ = map_msg
        #self.latest_map_ = map_msg

        # Extract basic info
        ox = map_msg.info.origin.position.x
        oy = map_msg.info.origin.position.y
        res = map_msg.info.resolution
        w   = map_msg.info.width
        h   = map_msg.info.height

        self.map_origin_xy_ = (ox, oy)
        self.map_resolution_ = res
        self.map_size_ = (w, h)

        # Useful for random-goal bounds (unchanged behaviour)
        self.xlim_ = [ox, ox + w*res]
        self.ylim_ = [oy, oy + h*res]

        # Convert to numpy grid (H,W) with values {-1,0,100}
        data = np.array(map_msg.data, dtype=np.int16).reshape(h, w)  # row-major
        self.map_grid_ = data

        # Initialise / resize visit map if needed
        if self.visit_counts_ is None or self.visit_counts_.shape != self.map_grid_.shape:
            self.visit_counts_ = np.zeros_like(self.map_grid_, dtype=np.float32)

    def test_yolo(self):
        img = cv2.imread("test.jpg")
        if img is None:
            print("❌ Failed to load image.")
            return

        # Optional resize to match training input
        img = cv2.resize(img, (640, 640))

        print("🚀 Running YOLO inference...")
        results = self.yolo_model_.predict(img, imgsz=640, conf=0.05, verbose=True)

        if not results or len(results[0].boxes) == 0:
            print("⚠️ No detections found.")
            return

        # Print detections
        for i, box in enumerate(results[0].boxes):
            xyxy = box.xyxy.cpu().numpy().flatten()
            conf = float(box.conf.cpu().numpy()[0])
            cls = int(box.cls.cpu().numpy()[0])
            label = f"{self.yolo_model_.names[cls]} {conf:.2f}"
            print(f"✅ Box {i}: {label}, coords=({xyxy[0]:.1f},{xyxy[1]:.1f},{xyxy[2]:.1f},{xyxy[3]:.1f})")

        # Save annotated result
        results[0].save(filename="test_output.jpg")
        print("💾 Saved annotated image as test_output.jpg")

    def image_callback(self, msg):
        # -------- params (keep here or move to __init__) --------
        if not hasattr(self, "allowed_classes"): self.allowed_classes = {"stop sign"}
        if not hasattr(self, "min_conf"):        self.min_conf = 0.5
        if not hasattr(self, "min_box_area"):    self.min_box_area = 600.0       # px^2
        if not hasattr(self, "max_box_frac"):    self.max_box_frac = 0.6         # fraction of frame
        if not hasattr(self, "edge_frac"):       self.edge_frac = 0.02
        if not hasattr(self, "hfov_deg"):        self.hfov_deg = 70.0            # camera HFOV (deg)
        if not hasattr(self, "go_to_standoff_on_detection"):
            self.go_to_standoff_on_detection = False  # set True to drive to standoff immediately

        cv_image = self.cv_bridge_.imgmsg_to_cv2(msg, "bgr8")
        self.last_image_ = cv_image
        H, W = cv_image.shape[:2]

        # YOLO
        results = self.yolo_model_.predict(cv_image, imgsz=640, conf=min(self.min_conf, 0.99), verbose=False)
        if not results:
            return
        det = results[0]

        # (optional) publish annotated image
        annotated = det.plot()
        self.image_detections_pub_.publish(self.cv_bridge_.cv2_to_imgmsg(annotated, "bgr8"))

        robot_pose = self.get_pose_2d()
        if robot_pose is None or self.map_grid_ is None or self.map_resolution_ is None:
            return

        hfov = math.radians(self.hfov_deg)

        for box in det.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            cls_name = self.yolo_model_.names.get(cls_id, str(cls_id))
            conf = float(box.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()

            # ---- filters ----
            if cls_name not in self.allowed_classes:
                continue
            if conf < self.min_conf:
                continue

            w = x2 - x1; h = y2 - y1; area = w * h
            if area < self.min_box_area:
                continue
            if area > self.max_box_frac * (W * H):
                continue  # huge box: probably robot/wall filling frame

            # edge filter
            mxg = self.edge_frac * W; myg = self.edge_frac * H
            if x1 < mxg or x2 > (W - mxg) or y1 < myg or y2 > (H - myg):
                continue

            # ---- image column -> bearing (simple pinhole) ----
            cx = 0.5 * (x1 + x2)
            bearing = (cx / W - 0.5) * hfov
            theta = wrap_angle(robot_pose.theta + bearing)

            # ---- raycast to the wall in the occupancy map ----
            hit = self._raycast_to_wall(robot_pose.x, robot_pose.y, theta,
                                        max_range_m=12.0,
                                        step_m=max(0.5*self.map_resolution_, 0.05))
            if hit is None:
                continue

            ax, ay, (mx, my) = hit  # artifact point on wall (world), and map index

            # (A) store artifact marker at the wall location
            p = Point(x=ax, y=ay, z=0.3)
            if not self.artifact_locations_ or math.hypot(
                p.x - self.artifact_locations_[-1].x,
                p.y - self.artifact_locations_[-1].y) > 0.3:
                self.artifact_locations_.append(p)
                self.publish_artifact_markers()

            # (B) compute a perpendicular 2 m standoff in free space
            nx, ny = self._wall_normal_from_occupancy(mx, my)  # unit normal into free space
            standoff = self._standoff_along_normal(ax, ay, nx, ny, dist_m=2.0, clearance_cells=3)
            if standoff is None:
                standoff = self._standoff_along_normal(ax, ay, nx, ny, dist_m=1.2, clearance_cells=3)

            if standoff is not None:
                sx, sy = standoff
                # Face the sign: heading from standoff → sign
                theta_goal = math.atan2(ay - sy, ax - sx)

                # Visualize standoff as your “frontier goal” marker (green sphere)
                self.marker_frontier_goal_.points = [Point(x=sx, y=sy, z=0.2)]
                arr = MarkerArray()
                arr.markers = [self.marker_frontiers_, self.marker_frontier_goal_]
                self.marker_pub_.publish(arr)

                # (optional) actually go there
                if self.go_to_standoff_on_detection and self.ready_for_next_goal_:
                    g = Pose2D(); g.x = sx; g.y = sy; g.theta = wrap_angle(theta_goal)
                    self.planner_go_to_pose2d(g)
                    self.ready_for_next_goal_ = False

    def localise_artifact(self):
        """
        Estimate and publish artifact location.
            Combines logic from Perception 1–2 and improved spatial placement.
        """

        # --- Old Perception 1–2 base logic (kept for report) ---
        # robot_pose = self.get_pose_2d()
        # if robot_pose is None:
        #     self.get_logger().warn('localise_artifact: robot_pose is None.')
        #     return
        # point = Point()
        # point.x = robot_pose.x
        # point.y = robot_pose.y
        # point.z = 1.0
        # self.artifact_locations_.append(point)
        # self.publish_artifact_markers()

        # --- New improved logic (for realistic localisation) ---
        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            return

        # Retrieve the last estimated artifact position (from image_callback)
        est_point = getattr(self, "last_estimated_artifact_", None)
        if est_point is None:
            self.get_logger().warn("⚠️ No estimated artifact position available yet.")
            return

        # Compute distance between robot and estimated artifact
        dist_to_est = math.hypot(est_point.x - robot_pose.x, est_point.y - robot_pose.y)

        # If the robot is close enough to the estimated artifact → publish marker slightly ahead
        if dist_to_est <= 0.5:
            forward_offset = 0.8  # distance ahead of robot to place the marker
            point = Point()
            point.x = robot_pose.x + forward_offset * math.cos(robot_pose.theta)
            point.y = robot_pose.y + forward_offset * math.sin(robot_pose.theta)
            point.z = 0.3

            # Only add if it's not a duplicate
            if not self.artifact_locations_ or math.hypot(
                point.x - self.artifact_locations_[-1].x,
                point.y - self.artifact_locations_[-1].y
            ) > 0.4:
                self.artifact_locations_.append(point)
                self.get_logger().info(
                    f"📍 Robot near artifact ({dist_to_est:.2f} m) — publishing marker ahead at ({point.x:.2f}, {point.y:.2f})"
                )
                self.publish_artifact_markers()
                self.get_logger().info(f"✅ Published {len(self.artifact_locations_)} artifact markers")

        else:
            self.get_logger().info(f"Artifact still far ({dist_to_est:.2f} m), not published yet.")

    def publish_artifact_markers(self):
        self.marker_artifacts_.points = self.artifact_locations_
        self.marker_artifacts_est_.points = self.artifact_estimates_

        arr = MarkerArray()
        arr.markers = [self.marker_artifacts_, self.marker_artifacts_est_]
        self.marker_pub_.publish(arr)

    def planner_go_to_pose2d(self, pose2d):
        """Go to a provided 2d pose"""

        # Send a goal to navigate_to_pose with self.nav2_action_client_
        action_goal = NavigateToPose.Goal()
        action_goal.pose.header.stamp = self.get_clock().now().to_msg()
        action_goal.pose.header.frame_id = 'map'
        action_goal.pose.pose = pose2d_to_pose(pose2d)

        # Publish visualisation
        self.goal_pose_vis_.publish(action_goal.pose)

        # Decide whether to show feedback or not
        if self.get_parameter('print_feedback').value:
            feedback_method = self.feedback_callback
        else:
            feedback_method = None

        # Send goal to action server
        self.get_logger().warn(f'Sending goal [{pose2d.x:.2f}, {pose2d.y:.2f}]...')
        self.send_goal_future_ = self.nav2_action_client_.send_goal_async(
            action_goal,
            feedback_callback=feedback_method)
        self.send_goal_future_.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """The requested goal pose has been sent to the action server"""

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            self.ready_for_next_goal_ = True
            return

        # Goal accepted: get result when it's completed
        self.get_logger().warn(f'Goal accepted')
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(self.goal_reached_callback)

    def feedback_callback(self, feedback_msg):
        """Monitor the feedback from the action server"""

        feedback = feedback_msg.feedback

        self.get_logger().info(f'{feedback.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future):
        """The requested goal has been reached"""

        result = future.result().result
        self.get_logger().info(f'Goal reached!')
        self.ready_for_next_goal_ = True

    def planner_move_forwards(self, distance):
        """Simply move forward by the specified distance"""

        pose_2d = self.get_pose_2d()

        pose_2d.x += distance * math.cos(pose_2d.theta)
        pose_2d.y += distance * math.sin(pose_2d.theta)

        self.planner_go_to_pose2d(pose_2d)

    def planner_go_to_first_artifact(self):
        """Go to a pre-specified artifact location"""

        goal_pose2d = Pose2D(
            x = 18.1,
            y = 6.6,
            theta = math.pi/2
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_return_home(self):
        """Return to the origin"""

        goal_pose2d = Pose2D(
            x = 0.0,
            y = 0.0,
            theta = math.pi
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_walk(self):
        """Go to a random location, which may be invalid"""

        # Select a random location
        goal_pose2d = Pose2D(
            x = random.uniform(self.xlim_[0], self.xlim_[1]),
            y = random.uniform(self.ylim_[0], self.ylim_[1]),
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_goal(self):
        """Go to a random location out of a predefined set"""

        # Hand picked set of goal locations
        random_goals = [[15.2, 2.2],
                        [30.7, 2.2],
                        [43.0, 11.3],
                        [36.6, 21.9],
                        [33.0, 30.4],
                        [40.4, 44.3],
                        [51.5, 37.8],
                        [16.0, 24.1],
                        [3.4, 33.5],
                        [7.9, 13.8],
                        [14.2, 37.7]]

        # Select a random location
        goal_valid = False
        while not goal_valid:
            idx = random.randint(0,len(random_goals)-1)
            goal_x = random_goals[idx][0]
            goal_y = random_goals[idx][1]

            # Only accept this goal if it's within the current costmap bounds
            if goal_x > self.xlim_[0] and goal_x < self.xlim_[1] and \
               goal_y > self.ylim_[0] and goal_y < self.ylim_[1]:
                goal_valid = True
            else:
                self.get_logger().warn(f'Goal [{goal_x}, {goal_y}] out of bounds')

        goal_pose2d = Pose2D(
            x = goal_x,
            y = goal_y,
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def main_loop(self):
        """
        Set the next goal pose and send to the action server
        See https://docs.nav2.org/concepts/index.html
        """
        
        # Don't do anything until SLAM is launched
        if not self.tf_buffer.can_transform(
                'map',
                'base_link',
                rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... Have you launched a SLAM node?')
            return

        ###########################################################################################
        # Keep a fading trail of where we've been
        self._decay_visit_counts()
        rp = self.get_pose_2d()
        if rp is not None:
            mm = self.world_to_map_index(rp.x, rp.y)
            if mm is not None:
                self._splat_visit(mm[0], mm[1], r_cells=self.kernel_radius_cells)
        ###########################################################################################


        # Update flags related to the progress of the current planner

        # Check if previous goal still running
        if not self.ready_for_next_goal_:
            # self.get_logger().info(f'Previous goal still running')
            return

        self.ready_for_next_goal_ = False

        if self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.get_logger().info('Successfully reached first artifact!')
            self.reached_first_artifact_ = True
        if self.planner_type_ == PlannerType.RETURN_HOME:
            self.get_logger().info('Successfully returned home!')
            self.returned_home_ = True

        # Select the next planner to execute
        # Update this logic as you see fit!
        if not self.reached_first_artifact_:
            self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
        elif not self.returned_home_:
            self.planner_type_ = PlannerType.RETURN_HOME
        else:
            ###########################################################################################
            self.planner_type_ = PlannerType.EXPLORE_FRONTIERS
            ###########################################################################################


        # Execute the planner by calling the relevant method
        # Add your own planners here!
        self.get_logger().info(f'Calling planner: {self.planner_type_.name}')
        if self.planner_type_ == PlannerType.MOVE_FORWARDS:
            self.planner_move_forwards(10)
        elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.planner_go_to_first_artifact()
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            self.planner_return_home()
        elif self.planner_type_ == PlannerType.RANDOM_WALK:
            self.planner_random_walk()
        elif self.planner_type_ == PlannerType.RANDOM_GOAL:
            self.planner_random_goal()
        ###########################################################################################
        elif self.planner_type_ == PlannerType.EXPLORE_FRONTIERS:
            self.planner_explore_frontiers()
        ###########################################################################################

        else:
            self.get_logger().error('No valid planner selected')
            self.destroy_node()

    def main_loop_perception(self):
        """Main control loop for autonomous dataset collection."""
        
        #new
        if hasattr(self, "last_goal_time_"):
            time_since_last_goal = (self.get_clock().now() -self.last_goal_time_).nanoseconds / 1e9
            if time_since_last_goal < 8.0:
                return
        self.last_goal_time_ = self.get_clock().now()
        
        # Wait until transform is available
        if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... (launch SLAM first)')
            return

        # Get current pose
        self.current_pose_ = self.get_pose_2d()
        if self.current_pose_ is None:
            return

        # --- Timed scan trigger ---
        time_since_last_scan = (self.get_clock().now() - self.last_scan_time_).nanoseconds / 1e9
        if time_since_last_scan > 10:  # Perform scan every ~20 s
            self.perform_scan_routine()
            self.last_scan_time_ = self.get_clock().now()
            return  # Wait for next cycle

        # --- Continue exploring randomly ---
        if getattr(self, 'ready_for_next_goal_', True):
            self.planner_random_goal()
            self.ready_for_next_goal_ = False
        else:
            # Wait until the current goal completes before selecting a new one
            return

#############################   PLANNING   #################################################
    
    def _neighbors8(self, mx, my):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx == 0 and dy == 0: 
                    continue
                yield (mx+dx, my+dy)
    
    def planner_explore_frontiers(self):
        """Compute frontiers -> choose goal -> send NavigateToPose."""
        if self.map_grid_ is None:
            self.get_logger().warn('No map grid yet; skipping frontier step.')
            self.ready_for_next_goal_ = True
            return

        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            self.ready_for_next_goal_ = True
            return

        fpix = self.find_frontier_pixels()
        clusters = self.cluster_frontiers(fpix)
        items = self.frontier_centroids_and_sizes(clusters)  # [(x,y,size), ...]
        goal = self.select_next_frontier_goal(items, robot_pose)

        # Visualise candidates & chosen goal
        self.publish_frontier_markers([(x, y) for (x, y, s) in items], goal)

        if goal is None:
            self.get_logger().info('No frontiers found — falling back to RANDOM_GOAL.')
            self.planner_random_goal()
            return

        self.planner_go_to_pose2d(goal)

    def find_frontier_pixels(self):
        """
        Frontier pixel: FREE (0) that touches at least one UNKNOWN (-1) in 8-neighborhood.
        Returns list of (mx,my) indices.
        """
        if self.map_grid_ is None: 
            return []

        grid = self.map_grid_
        h, w = grid.shape
        frontiers = []
        # skip 1px border to avoid bounds checks overhead
        for my in range(1, h-1):
            for mx in range(1, w-1):
                if grid[my, mx] == 0:  # free
                    neigh = [grid[ny, nx] for (nx, ny) in self._neighbors8(mx, my)]
                    if any(v == -1 for v in neigh):
                        frontiers.append((mx, my))
        return frontiers

    def cluster_frontiers(self, frontier_pixels):
        """
        Simple connected-component clustering on the frontier pixels using 8-connectivity.
        Returns list of clusters, each is a list of (mx,my).
        """
        if not frontier_pixels: 
            return []

        frontier_set = set(frontier_pixels)
        seen = set()
        clusters = []
        for seed in frontier_pixels:
            if seed in seen: 
                continue
            comp = []
            q = deque([seed])
            seen.add(seed)
            while q:
                p = q.popleft()
                comp.append(p)
                px, py = p
                for nx, ny in self._neighbors8(px, py):
                    if (nx, ny) in frontier_set and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        q.append((nx, ny))
            if len(comp) >= 12:  # was 5; bump it to ignore thread-like alcoves
                clusters.append(comp)
        return clusters

    def select_next_frontier_goal(self, items, robot_pose):
        """
        items: list of (cx, cy, size) in world coords.
        Score = + w_size*size                         (prefer informative/frontier size)
                - w_dist * d                          (prefer nearer)
                - w_visit_local * avg_visit_local     (avoid busy areas)
                - w_visit_path  * path_visit_integral (avoid reusing corridors)
        """
        if not items:
            return None

        rx, ry = robot_pose.x, robot_pose.y
        scored = []
        for (cx, cy, size) in items:
            d   = math.hypot(cx - rx, cy - ry)
            vloc = self._avg_visit_local(cx, cy, r_cells=3)
            vpath = self._path_visit_integral(rx, ry, cx, cy, step_m=0.2)

            # distance can be slightly convex to discourage long flips
            d_pen = d + 0.12 * d * d

            # optional info-gain term — counts how many unknown cells surround the frontier
            gain = self._info_gain_unknown(cx, cy, r_cells=6)

            score = (1.2 * gain                  # weight for information gain
                    + self.w_size * float(size) # bigger frontiers better
                    - self.w_dist * d_pen       # farther = worse
                    - self.w_visit_local * vloc # avoid revisiting
                    - self.w_visit_path  * vpath)

            scored.append((score, cx, cy, size, d, vloc, vpath))

        scored.sort(key=lambda t: t[0], reverse=True)

        for (_score, cx, cy, _size, _d, _vloc, _vpath) in scored:
            safe = self._backoff_to_clear_space(cx, cy, rx, ry,
                                                max_backoff_m=2.5,
                                                step_m=0.15,
                                                clearance_cells=3)
            if safe is None:
                continue
            gx, gy = safe
            theta = math.atan2(cy - ry, cx - rx)
            goal = Pose2D()
            goal.x, goal.y, goal.theta = gx, gy, wrap_angle(theta)
            return goal

        return None

    def publish_frontier_markers(self, centroids_world, chosen_goal):
        # candidate centroids
        pts = []
        for (x, y) in centroids_world:
            p = Point(x=x, y=y, z=0.1)
            pts.append(p)
        self.marker_frontiers_.points = pts

        # chosen goal
        if chosen_goal is not None:
            p = Point(x=chosen_goal.x, y=chosen_goal.y, z=0.2)
            self.marker_frontier_goal_.points = [p]
        else:
            self.marker_frontier_goal_.points = []

        # publish
        arr = MarkerArray()
        arr.markers = [self.marker_frontiers_, self.marker_frontier_goal_]
        self.marker_pub_.publish(arr)

    def world_to_map_index(self, x, y):
        """Return (mx,my) integer indices for map array, or None if outside."""
        if self.map_resolution_ is None: return None
        ox, oy = self.map_origin_xy_
        res = self.map_resolution_
        mx = int(np.floor((x - ox) / res))
        my = int(np.floor((y - oy) / res))
        w, h = self.map_size_
        if 0 <= mx < w and 0 <= my < h:
            return (mx, my)
        return None

    def map_index_to_world(self, mx, my):
        """Return (x,y) for centre of cell (mx,my)."""
        ox, oy = self.map_origin_xy_
        res = self.map_resolution_
        x = ox + (mx + 0.5) * res
        y = oy + (my + 0.5) * res
        return (x, y)    

    def _cell_free(self, mx, my):
        return (0 <= my < self.map_grid_.shape[0] and
                0 <= mx < self.map_grid_.shape[1] and
                self.map_grid_[my, mx] == 0)

    def _clearance_ok(self, mx, my, radius_cells=3):
        """Require a bubble of free cells around (mx,my)."""
        h, w = self.map_grid_.shape
        for dy in range(-radius_cells, radius_cells+1):
            for dx in range(-radius_cells, radius_cells+1):
                nx, ny = mx+dx, my+dy
                if not (0 <= ny < h and 0 <= nx < w): 
                    return False
                if self.map_grid_[ny, nx] != 0:
                    return False
        return True

    def _backoff_to_clear_space(self, gx, gy, rx, ry, max_backoff_m=2.0, step_m=0.15, clearance_cells=3):
        """
        Slide goal from frontier centroid back toward robot until
        there’s a free bubble of clearance_cells.
        """
        if self.map_resolution_ is None:
            return None
        ox, oy = self.map_origin_xy_
        res = self.map_resolution_

        # start at centroid
        vx, vy = gx - rx, gy - ry
        norm = math.hypot(vx, vy) or 1e-6
        ux, uy = vx/norm, vy/norm

        # march backwards in steps
        steps = int(max_backoff_m / step_m)
        for k in range(steps+1):
            x = gx - k * step_m * ux
            y = gy - k * step_m * uy
            mm = self.world_to_map_index(x, y)
            if mm is None:
                continue
            mx, my = mm
            if self._cell_free(mx, my) and self._clearance_ok(mx, my, radius_cells=clearance_cells):
                return (x, y)

        # fail -> None
        return None   

    def _avg_visit_local(self, xw, yw, r_cells=3):
        """Average visit count in a small window around a world point."""
        mm = self.world_to_map_index(xw, yw)
        if mm is None or self.visit_counts_ is None: 
            return 0.0
        mx, my = mm
        h, w = self.visit_counts_.shape
        y0 = max(0, my - r_cells); y1 = min(h-1, my + r_cells)
        x0 = max(0, mx - r_cells); x1 = min(w-1, mx + r_cells)
        patch = self.visit_counts_[y0:y1+1, x0:x1+1]
        if patch.size == 0: 
            return 0.0
        return float(np.mean(patch))

    def _path_visit_integral(self, x0, y0, x1, y1, step_m=0.15):
        """Approximate integral of visit counts along straight line (robot→centroid)."""
        if self.visit_counts_ is None or self.map_resolution_ is None:
            return 0.0
        dx, dy = (x1 - x0), (y1 - y0)
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return 0.0
        ux, uy = dx / L, dy / L
        samples = max(1, int(L / step_m))
        acc = 0.0
        for k in range(samples + 1):
            xs = x0 + k * step_m * ux
            ys = y0 + k * step_m * uy
            mm = self.world_to_map_index(xs, ys)
            if mm is None:
                continue
            mx, my = mm
            # clamp in bounds
            my = min(max(my, 0), self.visit_counts_.shape[0]-1)
            mx = min(max(mx, 0), self.visit_counts_.shape[1]-1)
            acc += self.visit_counts_[my, mx]
        # normalise a bit to keep scale tame
        return acc / float(samples + 1) 

    def _decay_visit_counts(self):
        if self.visit_counts_ is not None:
            np.multiply(self.visit_counts_, self.visit_decay_, out=self.visit_counts_)

    def _splat_visit(self, mx, my, r_cells=2):
        """Increment a small disc around (mx,my)."""
        if self.visit_counts_ is None: 
            return
        h, w = self.visit_counts_.shape
        r2 = r_cells * r_cells
        y0 = max(0, my - r_cells); y1 = min(h-1, my + r_cells)
        x0 = max(0, mx - r_cells); x1 = min(w-1, mx + r_cells)
        for y in range(y0, y1+1):
            dy2 = (y - my) * (y - my)
            for x in range(x0, x1+1):
                if (x - mx)*(x - mx) + dy2 <= r2 and self.map_grid_[y, x] == 0:
                    self.visit_counts_[y, x] += 1.0

    def frontier_centroids_and_sizes(self, clusters):
        """Return list of (x_world, y_world, size_in_pixels) for each cluster."""
        items = []
        for comp in clusters:
            if not comp:
                continue
            xs = [c[0] for c in comp]
            ys = [c[1] for c in comp]
            cx = int(round(sum(xs) / len(xs)))
            cy = int(round(sum(ys) / len(ys)))
            xw, yw = self.map_index_to_world(cx, cy)
            items.append((xw, yw, len(comp)))
        return items

    def _info_gain_unknown(self, xw, yw, r_cells=6):
        """Count unknown cells around (xw,yw)."""
        mm = self.world_to_map_index(xw, yw)
        if mm is None: 
            return 0.0
        mx, my = mm
        h, w = self.map_grid_.shape
        y0 = max(0, my - r_cells); y1 = min(h-1, my + r_cells)
        x0 = max(0, mx - r_cells); x1 = min(w-1, mx + r_cells)
        patch = self.map_grid_[y0:y1+1, x0:x1+1]
        return float(np.count_nonzero(patch == -1))

    def _raycast_to_frontier(self, rx, ry, theta, max_range_m=8.0, step_m=0.1):
        """March along a ray; return last free cell before hitting unknown/obstacle."""
        if self.map_grid_ is None or self.map_resolution_ is None:
            return None

        prev_free_xy = None
        steps = max(1, int(max_range_m / step_m))
        for k in range(1, steps + 1):
            x = rx + k * step_m * math.cos(theta)
            y = ry + k * step_m * math.sin(theta)
            mm = self.world_to_map_index(x, y)
            if mm is None:
                break
            mx, my = mm
            v = self.map_grid_[my, mx]
            if v == 0:
                prev_free_xy = (x, y)
                continue
            # if we hit unknown (-1) or obstacle (100), stand just before it
            if v == -1 or v == 100:
                return prev_free_xy
        return prev_free_xy

    def _raycast_to_wall(self, sx, sy, theta, max_range_m=12.0, step_m=None):
        """
        Cast a ray from (sx,sy) at heading theta through the occupancy grid.
        Returns (ax, ay, (mx, my)) where (mx,my) is the first occupied cell hit
        and (ax,ay) is its world coordinate (cell center).
        If no occupied cell is found within range, returns None.
        """
        if self.map_grid_ is None or self.map_resolution_ is None or self.map_origin_xy_ is None:
            return None

        res = self.map_resolution_
        step = step_m if step_m is not None else max(0.5*res, 0.05)

        ux, uy = math.cos(theta), math.sin(theta)
        t = 0.0
        prev_mm = None

        while t <= max_range_m:
            x = sx + t * ux
            y = sy + t * uy
            mm = self.world_to_map_index(x, y)
            if mm is None:
                # left the known map; stop
                return None

            mx, my = mm
            val = self.map_grid_[my, mx]

            if val == 100:  # occupied wall
                # Option A (default): place at the center of the hit occupied cell
                ax, ay = self.map_index_to_world(mx, my)

                # Option B: place at the boundary (last free before wall)
                # if prev_mm is not None:
                #     ax, ay = self.map_index_to_world(prev_mm[0], prev_mm[1])

                return (ax, ay, (mx, my))

            prev_mm = mm
            t += step

        return None

    def _wall_normal_from_occupancy(self, mx, my, win=3):
        """
        Estimate outward normal (into free space) at an occupied cell (mx,my)
        using a small gradient over a binary patch (occupied=1, other=0).
        Returns a unit vector (nx, ny). Fallback is (cos(theta_to_robot), sin(...)).
        """
        h, w = self.map_grid_.shape
        x0 = max(0, mx - win); x1 = min(w - 1, mx + win)
        y0 = max(0, my - win); y1 = min(h - 1, my + win)

        patch = (self.map_grid_[y0:y1+1, x0:x1+1] == 100).astype(np.float32)
        if patch.size < 9:  # too small, fallback
            return (1.0, 0.0)

        # simple sobel-like kernels
        kx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1,-2,-1],
                    [ 0, 0, 0],
                    [ 1, 2, 1]], dtype=np.float32)

        # pad and convolve
        P = np.pad(patch, 1, mode='edge')
        gx = cv2.filter2D(P, -1, kx)[1:-1,1:-1]
        gy = cv2.filter2D(P, -1, ky)[1:-1,1:-1]

        # take gradient at the patch center
        cy = (y1 - y0) // 2
        cx = (x1 - x0) // 2
        gxi = float(gx[cy, cx]); gyi = float(gy[cy, cx])

        n = np.array([gxi, gyi], dtype=np.float32)
        n_norm = float(np.linalg.norm(n)) or 1.0
        nx, ny = (n / n_norm).tolist()

        # ensure normal points into free space: sample a step along (nx,ny)
        test = self.world_to_map_index(*self.map_index_to_world(mx, my))
        if test is not None:
            # step 1 cell along the normal
            tx = int(round(mx + nx)); ty = int(round(my + ny))
            if 0 <= ty < h and 0 <= tx < w:
                if self.map_grid_[ty, tx] == 100:
                    # still occupied -> flip
                    nx, ny = -nx, -ny
        return (float(nx), float(ny))

    def _standoff_along_normal(self, ax, ay, nx, ny, dist_m=2.0, clearance_cells=3):
        """
        March from the wall point (ax,ay) along (nx,ny) up to dist_m.
        Return the first position (closest to target distance) that is free
        and has a clearance bubble of `clearance_cells`. If none, return None.
        """
        if self.map_resolution_ is None:
            return None

        step = max(0.5*self.map_resolution_, 0.05)
        steps = max(1, int(dist_m / step))

        best = None
        for k in range(steps, -1, -1):  # try far -> near so we prefer ~dist_m
            x = ax + k * step * nx
            y = ay + k * step * ny
            mm = self.world_to_map_index(x, y)
            if mm is None:
                continue
            mx, my = mm
            if self._cell_free(mx, my) and self._clearance_ok(mx, my, radius_cells=clearance_cells):
                best = (x, y)
                break
        return best

############################   PERCEPTION   #############################################

    def planner_rotate_in_place(self, n_angles=4):
        """Rotate in place to capture images from different angles."""
        current_pose = self.get_pose_2d()
        if current_pose is None:
            return

        for i in range(n_angles):
            # Rotate 360° evenly
            current_pose.theta = wrap_angle(current_pose.theta + 2 * math.pi / n_angles)
            self.planner_go_to_pose2d(current_pose)
            rclpy.spin_once(self, timeout_sec=2.0)  # wait briefly for pose update
            time.sleep(1.0)  # <— short pause for stability

    def perform_scan_routine(self):
        """Rotate and move slightly to collect diverse images automatically."""
        self.get_logger().info('Performing scan and shift...')

        # --- Get the current pose before rotating ---
        current_pose = self.get_pose_2d()
        if current_pose is None:
            return

        # --- Rotate and save images at each step ---
        n_angles = 4  # rotate 360° in 4 steps
        for i in range(n_angles):
            # Rotate 90 degrees each step
            current_pose.theta = wrap_angle(current_pose.theta + 2 * math.pi / n_angles)
            self.planner_go_to_pose2d(current_pose)
            self.get_logger().info(f'Rotating to angle {math.degrees(current_pose.theta):.1f}°')
            rclpy.spin_once(self, timeout_sec=2.0)  # allow time to reach pose

            # Capture and save the latest image if available
            if hasattr(self, "current_pose_") and self.current_pose_ is not None:
                last_image = getattr(self, "last_image_", None)
                if last_image is not None:
                    save_image(last_image, current_pose=self.current_pose_)

        # --- Move slightly forward for a new viewpoint ---
        self.planner_move_forwards(0.3)
        
        # New --- Allow next goal after scan ---
        self.ready_for_next_goal_ = True


def main():
    # Initialise
    rclpy.init()

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    while rclpy.ok():
        rclpy.spin(cave_explorer)