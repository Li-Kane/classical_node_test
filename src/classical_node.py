import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from std_msgs.msg import Bool
from tr_messages.msg import DetWithImg  # Adjust import based on your package structure
from cv_bridge import CvBridge
from rclpy.timer import Timer
from .decider import check_image, decide_to_shoot
from std_msgs.msg import Bool
import cv2

class ClassicalNode(Node):
    def __init__(self):
        super().__init__('classical_node')

        self.subscription = self.create_subscription(
            DetWithImg,
            'detections',
            self.listener_callback,
            10)
        
        self.shootpub = self.create_publisher(Bool, 'should_shoot', 10)

        self.count = 0
        
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.bridge = CvBridge()
        self.get_logger().info("ClassicalNode has been started.")
        self.opinions = []

    def listener_callback(self, msg: DetWithImg):

        # Convert ROS image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='bgr8')

        for det in msg.detection_info.detections:
            det: Detection2D
            conf = det.results[0].hypothesis.score
            bbox = det.bbox
            width = bbox.size_x
            height = bbox.size_y
            centerx = bbox.center.position.x
            centery = bbox.center.position.y

            x1 = int(centerx - (width / 2))
            y1 = int(centery - (height / 2))
            x2 = int(centerx + (width / 2))
            y2 = int(centery + (height / 2))

            # Crop the image
            cropped = cv_image[y1:y2, x1:x2,: ]
            if (self.count < 50):
                cv2.imwrite(f'images/image{self.count}.png', cropped)
                self.get_logger().info(f'count of {self.count}')
                self.count = self.count + 1
            
            opinion = check_image(cropped, conf, self.get_logger())
            self.opinions.append(opinion)

    def timer_callback(self):
        should_shoot = decide_to_shoot(self.opinions)
        boolmsg = Bool()
        boolmsg.data = should_shoot
        if should_shoot:
            self.shootpub.publish(boolmsg)
            self.get_logger().info("Shooting!")
        else:
            self.shootpub.publish(boolmsg)
            self.get_logger().info("Not Shooting!")
        
    
def main(args=None):
    rclpy.init(args=args)

    classical_node = ClassicalNode()

    rclpy.spin(classical_node)

    classical_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
