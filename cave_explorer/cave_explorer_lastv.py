
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

from cave_explorer.artifact_utils.save_images import save_image
import time

from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray

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
        """New map received, so update x and y limits"""
        self.latest_map_ = map_msg

        # Extract data from message
        map_origin = [map_msg.info.origin.position.x, 
                      map_msg.info.origin.position.y]
        map_resolution = map_msg.info.resolution
        map_height = map_msg.info.height
        map_width = map_msg.info.width

        # Set current limits
        self.xlim_ = [map_origin[0], map_origin[0]+map_width*map_resolution]
        self.ylim_ = [map_origin[1], map_origin[1]+map_height*map_resolution]

        # self.get_logger().warn('Map received:')
        # self.get_logger().warn(f'  xlim = [{self.xlim_[0]:.2f}, {self.xlim_[1]:.2f}]')
        # self.get_logger().warn(f'  ylim = [{self.ylim_[0]:.2f}, {self.ylim_[1]:.2f}]')
        
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
            self.publish_artifact_markers()
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


    def publish_artifact_markers(self):

        """Publish single Marker with multiple points (works fine)"""
        marker_array = MarkerArray()
        self.marker_artifacts_.points = self.artifact_locations_
        self.marker_artifacts_.header.stamp = self.get_clock().now().to_msg()
        marker_array.markers.append(self.marker_artifacts_)
        self.marker_pub_.publish(marker_array)
        self.get_logger().info(f"📍 Published {len(self.artifact_locations_)} markers to /marker_array_artifacts")


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

	#i think new
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
        """Main control loop for autonomous dataset collection."""
        
        # Allow faster re-goal selection, new incorporated to avoid robot to stuck in one place
        if hasattr(self, "last_goal_time_"):
            time_since_last_goal = (self.get_clock().now() - self.last_goal_time_).nanoseconds / 1e9
            if time_since_last_goal < 4.0:  # Reduced wait from 8s to 4s
                return
        else:
            time_since_last_goal = 0.0

        # Detect if robot is stuck (hasn't moved for 6s)
        if getattr(self, "last_pose_", None) is not None and self.current_pose_ is not None:
            dist_moved = math.hypot(
            self.current_pose_.x - self.last_pose_.x,
            self.current_pose_.y - self.last_pose_.y
            )
            if dist_moved < 0.15 and time_since_last_goal > 6.0:
                self.get_logger().warn("⚠️ Robot seems stuck. Picking new goal.")
                self.ready_for_next_goal_ = True
        else:
            # First loop or missing pose — just record the initial pose
            self.get_logger().debug("Initializing last_pose_ for movement tracking.")

        self.last_pose_ = self.current_pose_
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
        """Save artifact info (label and coordinates) to a simple text file."""
        try:
            filepath = "/home/student/ros_ws/src/cave_explorer/artifact_utils/artifact_locations_log.txt"
            with open(filepath, "a") as f:
                f.write(f"{label},{point.x:.2f},{point.y:.2f},{robot_pose.theta:.2f}\n")
            self.get_logger().info(f"💾 Saved artifact: {label} at ({point.x:.2f}, {point.y:.2f})")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to save artifact: {e}")





def main():
    rclpy.init()
    cave_explorer = CaveExplorer()
    rclpy.spin(cave_explorer)

        
        
