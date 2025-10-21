
import math
import random
from enum import Enum

import cv2  # OpenCV2
import rclpy
from pathlib import Path
import numpy as np
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

from cave_explorer.artifact_utils.save_images import save_image
import time

from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray


from collections import deque

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
    EXPLORE_FRONTIERS = 6   # <— NEW


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
        self.marker_artifacts_.scale.x = 1.5
        self.marker_artifacts_.scale.y = 1.5
        self.marker_artifacts_.scale.z = 1.5
        self.marker_artifacts_.color.a = 1.0
        self.marker_artifacts_.color.r = 0.0
        self.marker_artifacts_.color.g = 1.0
        self.marker_artifacts_.color.b = 0.2
        self.marker_pub_ = self.create_publisher(MarkerArray, '/marker_array_artifacts', 10)


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
        
        from std_msgs.msg import String
        self.artifact_info_pub_ = self.create_publisher(String, '/artifact_info', 10)



        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)
        
        self.last_scan_time_ = self.get_clock().now()  # Track when last scan occurred
        self.current_pose_ = None
        self.visit_counts_ = None  # lazily initialised once a map arrives
        self.last_inspect_trigger_time_ = None  # cooldown tracking for INSPECT mode
        self.coverage_threshold_ = 0.85
        self.latest_coverage_ratio_ = 0.0
        self.coverage_log_interval_sec_ = 15.0
        self.last_coverage_log_time_ = self.get_clock().now()
        self.no_frontier_cycles_ = 0
        self.frontier_hysteresis_ = 5
        self.exploration_complete_ = False
        self.reported_exploration_complete_ = False
        self.patrol_waypoints_ = []
        self.patrol_index_ = 0
        self.patrol_ready_ = False
        self.patrol_path_pub_ = self.create_publisher(Marker, 'patrol_path', 1)


        # --- YOLOv8 (Simplified Ultralytics API) ---
        try:
            self.yolo_model_ = YOLO('/home/student/ros_ws/src/cave_explorer-1/cave_explorer/models/artifacts/best.pt')
            self.get_logger().info('✅ Loaded YOLOv8 model from best.pt')
            self.get_logger().info(f"Model classes: {self.yolo_model_.names}")
        except Exception as e:
            self.get_logger().warn(f'⚠️ Failed to load YOLO model: {e}. Falling back to pretrained COCO model.')
            self.yolo_model_ = YOLO('yolov8n.pt')

        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model_.to(self.device)
        self.get_logger().info(f"🚀 YOLO model running on {self.device.upper()}")

        self.mode_ = "EXPLORE"                 # EXPLORE | INSPECT
        self.go_to_standoff_on_detection = True
        self.standoff_m = 2.0                  # target distance for close inspection
        self.last_estimated_artifact_ = None   # filled by image_callback

        # tuning weights
        self.w_size = 2.0     # prefer bigger frontier clusters
        self.w_dist = 1.0     # penalise distance
        self.w_visit_local = 1.5 # penalise locally revisited areas
        self.w_visit_path  = 1.4 # penalise path revisits

        self.marker_frontiers_ = Marker()
        self.marker_frontiers_.header.frame_id = "map"
        self.marker_frontiers_.ns = "frontiers"
        self.marker_frontiers_.type = Marker.POINTS
        self.marker_frontiers_.scale.x = self.marker_frontiers_.scale.y = 0.25
        self.marker_frontiers_.color.a = 1.0; self.marker_frontiers_.color.r = 1.0; self.marker_frontiers_.color.g = 1.0
        self.marker_frontier_goal_ = Marker()
        self.marker_frontier_goal_.header.frame_id = "map"
        self.marker_frontier_goal_.ns = "frontier_goal"
        self.marker_frontier_goal_.type = Marker.SPHERE_LIST
        self.marker_frontier_goal_.scale.x = self.marker_frontier_goal_.scale.y = self.marker_frontier_goal_.scale.z = 0.5
        self.marker_frontier_goal_.color.a = 1.0; self.marker_frontier_goal_.color.r = 0.2; self.marker_frontier_goal_.color.g = 0.8

        # Persistent artifact bookkeeping
        self.artifact_log_path_ = Path(__file__).resolve().parent / "artifact_utils" / "artifact_locations_log.txt"
        self.artifact_merge_threshold_ = 0.75  # metres
        self.artifact_records_ = []
        self.consolidated_artifacts_ = []
        try:
            self.artifact_log_path_.parent.mkdir(parents=True, exist_ok=True)
            if self.artifact_log_path_.exists():
                self.artifact_log_path_.write_text("")
                self.get_logger().info("🧹 Cleared artifact log at startup.")
        except Exception as exc:
            self.get_logger().warn(f"Could not clear artifact log: {exc}")
        self.load_artifact_log()
        self.publish_artifact_markers()

    
    def get_pose_2d(self):
        """Get the 2d pose of the robot"""
        # Check transform availability before lookup
        if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
            self.get_logger().warn('Waiting for transform (map→base_link)...')
            return None

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
        # Keep original bounds for your random goals
        self.latest_map_ = map_msg
        ox = map_msg.info.origin.position.x
        oy = map_msg.info.origin.position.y
        res = map_msg.info.resolution
        w   = map_msg.info.width
        h   = map_msg.info.height

        self.xlim_ = [ox, ox + w*res]
        self.ylim_ = [oy, oy + h*res]

        # --- NEW: cache full map for frontier detection (Planning 1) ---
        import numpy as np
        self.map_origin_xy_ = (ox, oy)
        self.map_resolution_ = res
        self.map_size_ = (w, h)
        self.map_grid_ = np.array(map_msg.data, dtype=np.int16).reshape(h, w)  # -1 unknown, 0 free, 100 occ

        # visit “heatmap” for revisit-avoidance
        if not hasattr(self, "visit_counts_") or self.visit_counts_ is None or self.visit_counts_.shape != self.map_grid_.shape:
            self.visit_counts_ = np.zeros_like(self.map_grid_, dtype=np.float32)
        
    def test_yolo(self):
        img = cv2.imread("test.jpg")
        if img is None:
            print(" Failed to load image.")
            return

        # Optional resize to match training input
        img = cv2.resize(img, (640, 640))

        print(" Running YOLO inference...")
        results = self.yolo_model_.predict(img, imgsz=640, conf=0.05, verbose=True)

        if not results or len(results[0].boxes) == 0:
            print(" No detections found.")
            return

        # Print detections
        for i, box in enumerate(results[0].boxes):
            xyxy = box.xyxy.cpu().numpy().flatten()
            conf = float(box.conf.cpu().numpy()[0])
            cls = int(box.cls.cpu().numpy()[0])
            label = f"{self.yolo_model_.names[cls]} {conf:.2f}"
            print(f" Box {i}: {label}, coords=({xyxy[0]:.1f},{xyxy[1]:.1f},{xyxy[2]:.1f},{xyxy[3]:.1f})")

        # Save annotated result
        results[0].save(filename="test_output.jpg")
        print(" Saved annotated image as test_output.jpg")
    
    def image_callback(self, msg):
        """Process each camera frame and detect artifacts"""
        cv_image = self.cv_bridge_.imgmsg_to_cv2(msg, "bgr8")
        self.last_image_ = cv_image

        # Run YOLO detection
        results = self.yolo_model_.predict(cv_image, imgsz=640, conf=0.3)
        if not results or len(results[0].boxes) == 0:
            return
            
        # New Correct labels to save location
        if len(results[0].boxes) > 0:
            cls = int(results[0].boxes[0].cls.cpu().numpy()[0])
            self.last_detected_class_ = self.yolo_model_.names[cls]
            self.get_logger().info(f"🎯 Detected artifact: {self.last_detected_class_}")

        # Draw boxes or publish annotated image if you like
        annotated_frame = results[0].plot()
        self.image_detections_pub_.publish(
            self.cv_bridge_.cv2_to_imgmsg(annotated_frame, "bgr8")
        )

        # --- Estimate artifact direction (for localisation) ---
        robot_pose = self.get_pose_2d()
        if robot_pose:
            est_point = Point()
            est_point.x = robot_pose.x + 1.0 * math.cos(robot_pose.theta)
            est_point.y = robot_pose.y + 1.0 * math.sin(robot_pose.theta)
            est_point.z = 0.3
            self.last_estimated_artifact_ = est_point
            self.get_logger().info(
                f"🧭 Estimated artifact ahead at ({est_point.x:.2f}, {est_point.y:.2f})"
            )

            if self.go_to_standoff_on_detection:
                now = self.get_clock().now()
                if (
                    self.last_inspect_trigger_time_ is None
                    or (now - self.last_inspect_trigger_time_).nanoseconds > 3e9
                ):
                    self.mode_ = "INSPECT"   # pause exploration and do Planning 2 on next main_loop tick
                    self.last_inspect_trigger_time_ = now

        # Trigger localisation step (optional)
        self.localise_artifact()


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
            self.get_logger().warn(" No estimated artifact position available yet.")
            return
            
        dx = est_point.x - robot_pose.x
        dy = est_point.y - robot_pose.y
        angle_to_artifact = math.atan2(dy, dx)
        robot_pose.theta = angle_to_artifact
        self.get_logger().info(f"🔄 Rotating to face artifact (θ={math.degrees(angle_to_artifact):.1f}°)")

        # Compute distance between robot and estimated artifact
        dist_to_est = math.hypot(est_point.x - robot_pose.x, est_point.y - robot_pose.y)

        # Compute distance to the last published marker (avoid trails)
        if self.artifact_locations_:
            last_marker = self.artifact_locations_[-1]
            dist_from_last_marker = math.hypot(
                robot_pose.x - last_marker.x,
                robot_pose.y - last_marker.y
            )
        else:
            dist_from_last_marker = float('inf')

        # Only place marker if close to artifact AND far enough from last marker
        now = self.get_clock().now()
        cooldown_ok = not hasattr(self, "last_marker_time_") or \
              (now - self.last_marker_time_).nanoseconds / 1e9 > 6.0

        if dist_to_est <= 1.0 and dist_from_last_marker > 2.0 and cooldown_ok:
            # Position marker slightly in front of the robot
            forward_offset = 0.6 + random.uniform(0.1, 0.3)
            point = Point()
            point.x = robot_pose.x + forward_offset * math.cos(robot_pose.theta)
            point.y = robot_pose.y + forward_offset * math.sin(robot_pose.theta)
            point.z = 0.3

            # Save and publish
            self.artifact_locations_.append(point)
            self.get_logger().info(
                f"📍 Placed artifact marker at ({point.x:.2f}, {point.y:.2f}) "
                f"(dist_to_est={dist_to_est:.2f} m)"
            )

            # Send coordinates for Planning 2
            from std_msgs.msg import String
            msg = String()
            msg.data = f"{point.x:.2f},{point.y:.2f},{robot_pose.theta:.2f}"
            self.artifact_info_pub_.publish(msg)
            self.get_logger().info(f"✅ Published artifact goal for Planning 2: {msg.data}")
            
            #New save artifact estimate position
            label = getattr(self, "last_detected_class_", "Unknown")
            self.save_artifact_to_file(label, point, robot_pose)
            self.publish_artifact_markers()

            # Cooldown timer to avoid re-publishing markers too soon
            self.last_marker_time_ = self.get_clock().now()

        else:
            # Optional: only move closer if not yet near enough
            if dist_to_est > 1.0 and self.ready_for_next_goal_:
                self.ready_for_next_goal_ = False  # avoid spamming goals

                goal_pose = Pose2D()
                goal_pose.x = robot_pose.x + 1.5 * math.cos(robot_pose.theta)
                goal_pose.y = robot_pose.y + 1.5 * math.sin(robot_pose.theta)
                goal_pose.theta = robot_pose.theta

                self.planner_go_to_pose2d(goal_pose)
                self.get_logger().info(
                    f"🚶 Moving closer to artifact (currently {dist_to_est:.2f} m away)"
                )


    def calculate_coverage_ratio(self):
        """Return proportion of known free space that has been traversed."""
        if self.map_grid_ is None or self.visit_counts_ is None:
            return None
        free_mask = (self.map_grid_ == 0)
        total_free = np.count_nonzero(free_mask)
        if total_free == 0:
            return None
        visited_mask = (self.visit_counts_ > 0.05)
        visited_free = np.count_nonzero(np.logical_and(free_mask, visited_mask))
        return visited_free / total_free

    def _update_exploration_status(self, frontier_items):
        """Track coverage progress and detect when exploration is effectively complete."""
        coverage = self.calculate_coverage_ratio()
        if coverage is not None:
            self.latest_coverage_ratio_ = coverage
            now = self.get_clock().now()
            if (now - self.last_coverage_log_time_).nanoseconds / 1e9 > self.coverage_log_interval_sec_:
                self.last_coverage_log_time_ = now
                self.get_logger().info(f"📊 Coverage progress: {coverage*100:.1f}% of mapped free space visited.")

        has_frontiers = bool(frontier_items)
        if has_frontiers:
            self.no_frontier_cycles_ = 0
        else:
            self.no_frontier_cycles_ += 1

        if (
            not has_frontiers and
            coverage is not None and
            coverage >= self.coverage_threshold_ and
            self.no_frontier_cycles_ >= self.frontier_hysteresis_ and
            not self.exploration_complete_
        ):
            self.exploration_complete_ = True
            self.get_logger().info(f"✅ Exploration complete detected (coverage {coverage*100:.1f}%).")


    def publish_artifact_markers(self):
        """Publish consolidated artifact markers with per-class shapes and colours."""
        stamp = self.get_clock().now().to_msg()

        # Ensure consolidation is up to date if we loaded from disk only
        if not self.consolidated_artifacts_ and self.artifact_log_path_.exists():
            self.load_artifact_log()

        marker_array = MarkerArray()
        for idx, entry in enumerate(self.consolidated_artifacts_):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "artifact_shapes"
            marker.id = idx
            marker.type = self._marker_type_for_label(entry['label'])
            marker.action = Marker.ADD
            marker.pose.position.x = entry['x']
            marker.pose.position.y = entry['y']
            marker.pose.position.z = 0.3
            yaw = entry.get('theta', 0.0)
            marker.pose.orientation.z = math.sin(yaw / 2.0)
            marker.pose.orientation.w = math.cos(yaw / 2.0)
            marker.scale.x = marker.scale.y = 0.6
            marker.scale.z = 0.4
            r, g, b = self._marker_color_for_label(entry['label'])
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        self.marker_pub_.publish(marker_array)
        self.get_logger().info(
            f"📍 Published {len(self.consolidated_artifacts_)} consolidated markers to /marker_array_artifacts"
        )


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

        # Goal accepted: get result when it's completed, new
        self.get_logger().info('Goal accepted, waiting for completion...')
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(self.goal_reached_callback)

    def feedback_callback(self, feedback_msg):
        """Monitor the feedback from the action server"""

        feedback = feedback_msg.feedback

        self.get_logger().info(f'{feedback.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future):
        """The requested goal has been reached"""
        
        #Called when the goal is actually reached to avoid jumping

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
            if hasattr(self, "last_goal_idx") and abs(idx - self.last_goal_idx) <= 1:
                idx = (idx + 3) % len(random_goals)  # small skip to avoid repetition
            self.last_goal_idx = idx
            goal_x = random_goals[idx][0]
            goal_y = random_goals[idx][1]
            
            #new
        
            if self.current_pose_:
                 dist = math.hypot(goal_x - self.current_pose_.x, goal_y - self.current_pose_.y)
                 if dist < 5.0:
                    self.get_logger().warn("Goal too close, picking another...")
                    continue

            # New, Only accept this goal if it's within the current costmap bounds
            if self.xlim_[0] < goal_x < self.xlim_[1] and self.ylim_[0] < goal_y < self.ylim_[1]: 
                goal_valid = True
            else:
                self.get_logger().warn(f"Goal [{goal_x}, {goal_y}] out of bounds")

        goal_pose2d = Pose2D(x=goal_x, y=goal_y, theta=random.uniform(0, 2 * math.pi))
        self.planner_go_to_pose2d(goal_pose2d)

    def main_loop_original(self):
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

        #######################################################
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

        #######################################################
        # Select the next planner to execute
        # Update this logic as you see fit!
        if not self.reached_first_artifact_:
            self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
        elif not self.returned_home_:
            self.planner_type_ = PlannerType.RETURN_HOME
        else:
            self.planner_type_ = PlannerType.RANDOM_GOAL

        #######################################################
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
        else:
            self.get_logger().error('No valid planner selected')
            self.destroy_node()


        #######################################################
        
    def main_loop(self):
        # Wait for SLAM transform
        if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... (launch SLAM first)')
            return

        # keep “trail” (optional — avoids reusing same corridors)
        self._decay_visit_counts()
        rp = self.get_pose_2d()
        self.current_pose_ = rp  # keep this updated for distance checks
        if rp is not None:
            mm = self.world_to_map_index(rp.x, rp.y)
            if mm is not None:
                self._splat_visit(mm[0], mm[1], r_cells=2)

        # If we’re already executing a goal, chill
        if not self.ready_for_next_goal_:
            return

        # Simple intro sequence (same as before)
        if not self.reached_first_artifact_:
            self.ready_for_next_goal_ = False
            self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
            self.planner_go_to_first_artifact()
            self.reached_first_artifact_ = True
            return
        if not self.returned_home_:
            self.ready_for_next_goal_ = False
            self.planner_type_ = PlannerType.RETURN_HOME
            self.planner_return_home()
            self.returned_home_ = True
            return

        # --- Behaviour gate: INSPECT vs EXPLORE ---
        if self.mode_ == "INSPECT" and self.last_estimated_artifact_ is not None:
            self.ready_for_next_goal_ = False
            # Build a standoff goal that FACES the artifact (Planning 2)
            ax, ay = self.last_estimated_artifact_.x, self.last_estimated_artifact_.y
            rx, ry, rth = rp.x, rp.y, rp.theta
            import math
            theta_to_art = math.atan2(ay - ry, ax - rx)
            sx = ax - self.standoff_m * math.cos(theta_to_art)
            sy = ay - self.standoff_m * math.sin(theta_to_art)
            goal = Pose2D(x=sx, y=sy, theta=wrap_angle(theta_to_art))  # face the artifact
            self.get_logger().info(f'Planning 2: standoff -> ({sx:.2f},{sy:.2f}) face θ={math.degrees(theta_to_art):.1f}°')
            self.planner_go_to_pose2d(goal)
            # After we send one inspection goal, end mission here (Planning 2’s spec),
            # or flip back to exploration if you want to continue automatically:
            self.mode_ = "EXPLORE"
            self.last_estimated_artifact_ = None  # optional: prevents reuse of the old point
            return

        # Otherwise EXPLORE via frontiers (Planning 1)
        self.ready_for_next_goal_ = False
        self.planner_type_ = PlannerType.EXPLORE_FRONTIERS
        self.planner_explore_frontiers()
        

            
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
       
       
    #New to save coordinates for Planning 2 testing    
    def save_artifact_to_file(self, label, point, robot_pose):
        """Persist artifact info (label and coordinates) to disk and update cache."""
        try:
            self.artifact_log_path_.parent.mkdir(parents=True, exist_ok=True)
            with self.artifact_log_path_.open("a") as f:
                f.write(f"{label},{point.x:.2f},{point.y:.2f},{robot_pose.theta:.2f}\n")
            self._add_artifact_record(label, point.x, point.y, robot_pose.theta)
            self.get_logger().info(f"💾 Saved artifact: {label} at ({point.x:.2f}, {point.y:.2f})")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to save artifact: {e}")

    def load_artifact_log(self):
        """Load artifact observations from disk and consolidate nearby entries."""
        try:
            self.artifact_log_path_.parent.mkdir(parents=True, exist_ok=True)
            if not self.artifact_log_path_.exists():
                self.artifact_records_ = []
                self.consolidated_artifacts_ = []
                return

            records = []
            with self.artifact_log_path_.open("r") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 4:
                        self.get_logger().warn(f"Skipping malformed artifact log line: {line}")
                        continue
                    label = parts[0]
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        theta = float(parts[3])
                    except ValueError:
                        self.get_logger().warn(f"Skipping artifact with invalid coordinates: {line}")
                        continue
                    records.append({"label": label, "x": x, "y": y, "theta": theta})

            self.artifact_records_ = records
            self._recompute_consolidated_artifacts()
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load artifact log: {e}")
            self.artifact_records_ = []
            self.consolidated_artifacts_ = []

    def _add_artifact_record(self, label, x, y, theta):
        """Add a single artifact record and refresh consolidation."""
        self.artifact_records_.append({"label": label, "x": x, "y": y, "theta": theta})
        self._recompute_consolidated_artifacts()

    def _recompute_consolidated_artifacts(self):
        """Cluster artifact records of the same label that lie within the merge threshold."""
        clusters = []
        for record in self.artifact_records_:
            matched = None
            for cluster in clusters:
                if cluster["label"] != record["label"]:
                    continue
                cx = cluster["x_sum"] / cluster["count"]
                cy = cluster["y_sum"] / cluster["count"]
                if math.hypot(record["x"] - cx, record["y"] - cy) <= self.artifact_merge_threshold_:
                    matched = cluster
                    break

            if matched:
                matched["x_sum"] += record["x"]
                matched["y_sum"] += record["y"]
                matched["theta_sum"] += record["theta"]
                matched["count"] += 1
            else:
                clusters.append({
                    "label": record["label"],
                    "x_sum": record["x"],
                    "y_sum": record["y"],
                    "theta_sum": record["theta"],
                    "count": 1,
                })

        consolidated = []
        for cluster in clusters:
            count = cluster["count"]
            if count == 0:
                continue
            consolidated.append({
                "label": cluster["label"],
                "x": cluster["x_sum"] / count,
                "y": cluster["y_sum"] / count,
                "theta": cluster["theta_sum"] / count,
            })

        self.consolidated_artifacts_ = consolidated

    def _marker_type_for_label(self, label):
        """Map artifact label to an RViz marker type."""
        squares = {"Diamon", "Stop_Sign", "Ice"}
        return Marker.CUBE if label in squares else Marker.SPHERE

    def _marker_color_for_label(self, label):
        """Return RGB tuple for each artifact label."""
        palette = {
            "Diamon": (0.0, 0.8, 0.0),      # green square
            "Circle": (0.6, 0.6, 0.6),      # grey circle
            "Mushroom": (0.65, 0.3, 0.8),   # purple circle
            "Stop_Sign": (1.0, 0.1, 0.1),   # red square
            "Ice": (0.9, 0.9, 0.9),         # white square
            "Alien": (1.0, 0.55, 0.0),      # orange circle
        }
        return palette.get(label, (0.2, 0.2, 1.0))

    # ---- Frontier helpers (from Planning 1) ----
    def world_to_map_index(self, x, y):
        if self.map_resolution_ is None: return None
        ox, oy = self.map_origin_xy_
        res = self.map_resolution_
        mx = int((x - ox) // res)
        my = int((y - oy) // res)
        w, h = self.map_size_
        return (mx, my) if 0 <= mx < w and 0 <= my < h else None

    def map_index_to_world(self, mx, my):
        ox, oy = self.map_origin_xy_
        res = self.map_resolution_
        return (ox + (mx + 0.5)*res, oy + (my + 0.5)*res)

    def _neighbors8(self, mx, my):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx or dy:
                    yield (mx+dx, my+dy)

    def _cell_free(self, mx, my):
        return (0 <= my < self.map_grid_.shape[0] and
                0 <= mx < self.map_grid_.shape[1] and
                self.map_grid_[my, mx] == 0)

    def _clearance_ok(self, mx, my, radius_cells=3):
        h, w = self.map_grid_.shape
        for dy in range(-radius_cells, radius_cells+1):
            for dx in range(-radius_cells, radius_cells+1):
                nx, ny = mx+dx, my+dy
                if not (0 <= ny < h and 0 <= nx < w): return False
                if self.map_grid_[ny, nx] != 0:       return False
        return True

    def find_frontier_pixels(self):
        if self.map_grid_ is None: return []
        grid = self.map_grid_
        h, w = grid.shape
        frontiers = []
        for my in range(1, h-1):
            for mx in range(1, w-1):
                if grid[my, mx] == 0:
                    if any(grid[ny, nx] == -1 for (nx, ny) in self._neighbors8(mx, my)):
                        frontiers.append((mx, my))
        return frontiers

    def cluster_frontiers(self, frontier_pixels):
        if not frontier_pixels: return []
        fs = set(frontier_pixels); seen = set(); clusters = []
        for seed in frontier_pixels:
            if seed in seen: continue
            q = deque([seed]); seen.add(seed); comp = []
            while q:
                px, py = q.popleft(); comp.append((px,py))
                for nx, ny in self._neighbors8(px, py):
                    if (nx, ny) in fs and (nx, ny) not in seen:
                        seen.add((nx, ny)); q.append((nx, ny))
            if len(comp) >= 12: clusters.append(comp)  # ignore tiny wisps
        return clusters

    def frontier_centroids_and_sizes(self, clusters):
        items = []
        for comp in clusters:
            xs = [c[0] for c in comp]; ys = [c[1] for c in comp]
            cx, cy = int(round(sum(xs)/len(xs))), int(round(sum(ys)/len(ys)))
            xw, yw = self.map_index_to_world(cx, cy)
            items.append((xw, yw, len(comp)))
        return items

    def _info_gain_unknown(self, xw, yw, r_cells=6):
        mm = self.world_to_map_index(xw, yw)
        if mm is None: return 0.0
        import numpy as np
        mx, my = mm; h, w = self.map_grid_.shape
        y0, y1 = max(0,my-r_cells), min(h-1,my+r_cells)
        x0, x1 = max(0,mx-r_cells), min(w-1,mx+r_cells)
        patch = self.map_grid_[y0:y1+1, x0:x1+1]
        return float(np.count_nonzero(patch == -1))

    # visit “heatmap” terms (revisit-avoidance)
    def _decay_visit_counts(self):
        import numpy as np
        if self.visit_counts_ is not None:
            np.multiply(self.visit_counts_, 0.997, out=self.visit_counts_)  # light decay

    def _splat_visit(self, mx, my, r_cells=2):
        if self.visit_counts_ is None: return
        h, w = self.visit_counts_.shape; r2 = r_cells*r_cells
        for y in range(max(0,my-r_cells), min(h-1,my+r_cells)+1):
            dy2 = (y - my)*(y - my)
            for x in range(max(0,mx-r_cells), min(w-1,mx+r_cells)+1):
                if (x-mx)*(x-mx) + dy2 <= r2 and self.map_grid_[y, x] == 0:
                    self.visit_counts_[y, x] += 1.0

    def _avg_visit_local(self, xw, yw, r_cells=3):
        mm = self.world_to_map_index(xw, yw)
        if mm is None or self.visit_counts_ is None: return 0.0
        import numpy as np
        mx, my = mm; h, w = self.visit_counts_.shape
        y0, y1 = max(0,my-r_cells), min(h-1,my+r_cells)
        x0, x1 = max(0,mx-r_cells), min(w-1,mx+r_cells)
        patch = self.visit_counts_[y0:y1+1, x0:x1+1]
        return float(np.mean(patch)) if patch.size else 0.0

    def _path_visit_integral(self, x0, y0, x1, y1, step_m=0.15):
        if self.visit_counts_ is None or self.map_resolution_ is None: return 0.0
        import math
        dx, dy = (x1-x0), (y1-y0); L = math.hypot(dx, dy)
        if L < 1e-6: return 0.0
        ux, uy = dx/L, dy/L; samples = max(1, int(L/step_m)); acc = 0.0
        for k in range(samples+1):
            xs, ys = x0 + k*step_m*ux, y0 + k*step_m*uy
            mm = self.world_to_map_index(xs, ys)
            if mm is None: continue
            mx, my = mm
            my = min(max(my,0), self.visit_counts_.shape[0]-1)
            mx = min(max(mx,0), self.visit_counts_.shape[1]-1)
            acc += self.visit_counts_[my, mx]
        return acc / float(samples + 1)

    def _backoff_to_clear_space(self, gx, gy, rx, ry, max_backoff_m=2.5, step_m=0.15, clearance_cells=3):
        import math
        if self.map_resolution_ is None: return None
        vx, vy = gx-rx, gy-ry; n = math.hypot(vx, vy) or 1e-6
        ux, uy = vx/n, vy/n
        steps = int(max_backoff_m / step_m)
        for k in range(steps+1):
            x = gx - k*step_m*ux; y = gy - k*step_m*uy
            mm = self.world_to_map_index(x, y)
            if mm is None: continue
            mx, my = mm
            if self._cell_free(mx, my) and self._clearance_ok(mx, my, radius_cells=clearance_cells):
                return (x, y)
        return None

    # Goal Selection and Planner

    def publish_frontier_markers(self, centroids_world, chosen_goal):
        from geometry_msgs.msg import Point
        pts = [Point(x=x, y=y, z=0.1) for (x, y) in centroids_world]
        self.marker_frontiers_.points = pts
        if chosen_goal:
            self.marker_frontier_goal_.points = [Point(x=chosen_goal.x, y=chosen_goal.y, z=0.2)]
        else:
            self.marker_frontier_goal_.points = []
        arr = MarkerArray()
        arr.markers = [self.marker_frontiers_, self.marker_frontier_goal_]
        self.marker_pub_.publish(arr)

    def select_next_frontier_goal(self, items, robot_pose):
        import math
        if not items: return None
        rx, ry = robot_pose.x, robot_pose.y
        scored = []
        for (cx, cy, size) in items:
            d = math.hypot(cx-rx, cy-ry)
            d_pen = d + 0.12*d*d
            gain = self._info_gain_unknown(cx, cy, r_cells=6)
            vloc = self._avg_visit_local(cx, cy, r_cells=3)
            vpath = self._path_visit_integral(rx, ry, cx, cy, step_m=0.2)
            score = (1.2*gain + self.w_size*size - self.w_dist*d_pen - self.w_visit_local*vloc - self.w_visit_path*vpath)
            scored.append((score, cx, cy))
        scored.sort(key=lambda t: t[0], reverse=True)
        for _s, cx, cy in scored:
            safe = self._backoff_to_clear_space(cx, cy, rx, ry, clearance_cells=3)
            if safe is None: continue
            gx, gy = safe; theta = math.atan2(cy-ry, cx-rx)
            goal = Pose2D(); goal.x, goal.y, goal.theta = gx, gy, wrap_angle(theta)
            return goal
        return None

    def planner_explore_frontiers(self):
        if self.map_grid_ is None:
            self.get_logger().warn('No map yet; skipping frontier step.')
            self.ready_for_next_goal_ = True
            return
        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            self.ready_for_next_goal_ = True
            return
        fpix = self.find_frontier_pixels()
        clusters = self.cluster_frontiers(fpix)
        items = self.frontier_centroids_and_sizes(clusters)
        goal = self.select_next_frontier_goal(items, robot_pose)
        self.publish_frontier_markers([(x, y) for (x, y, _s) in items], goal)

        self._update_exploration_status(items)
        if self.exploration_complete_:
            self.ready_for_next_goal_ = True
            if not self.reported_exploration_complete_:
                self.reported_exploration_complete_ = True
                self.get_logger().info(
                    f"🏁 Exploration goal satisfied; ready for persistent monitoring. "
                    f"Coverage={self.latest_coverage_ratio_*100:.1f}%"
                )
            return

        if goal is None:
            self.planner_random_goal()  # fallback
            return
        self.planner_go_to_pose2d(goal)

def main():
    rclpy.init()
    cave_explorer = CaveExplorer()
    rclpy.spin(cave_explorer)

        
        
