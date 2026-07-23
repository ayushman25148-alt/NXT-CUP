# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors

QOS_PROFILE_DEFAULT = 10
PI = math.pi

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)

# HINT: You can adjust what percentage of the image from the bottom is analyzed.
# Lower portions are closer to the buggy, while upper portions see further ahead.
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.225
VECTOR_MAGNITUDE_MINIMUM = 2.25

# ---- HSV/LAB thresholding tuning knobs ----
# "Black" here means: dark (low Value / low L*) AND not strongly colorful (low Saturation).
# Using V alone (grayscale-equivalent) is very sensitive to global brightness changes.
# Adding an S ceiling helps reject colored shadows/reflections that a plain gray
# threshold would misclassify as "dark enough" or "not dark enough".
HSV_VALUE_MAX = 60          # V channel: how dark a pixel must be to count as "black" (0-255)
HSV_SATURATION_MAX = 90     # S channel: reject saturated/colored pixels even if dark
LAB_L_MAX = 65              # L* channel from LAB: secondary lightness check (0-255 in OpenCV's 8-bit LAB)
MORPH_KERNEL_SIZE = 3       # cleans up speckle noise in the binary mask


class EdgeVectorsPublisher(Node):
    """
    ROS 2 Node that processes raw camera images to detect the lane edges (left/right bounds).
    It publishes the detected boundaries as synapse_msgs/EdgeVectors.
    """
    def __init__(self):
        super().__init__('edge_vectors_publisher')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        # Publisher for edge vectors.
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT)

        # Publisher for thresh image (for debugging thresholding/segmentation).
        self.publisher_thresh_image = self.create_publisher(
            CompressedImage,
            "/debug_images/thresh_image",
            QOS_PROFILE_DEFAULT)

        # Publisher for vector image (for debugging vector drawing).
        self.publisher_vector_image = self.create_publisher(
            CompressedImage,
            "/debug_images/vector_image",
            QOS_PROFILE_DEFAULT)

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

        # Reusable morphology kernel for cleaning the binary mask.
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    def publish_debug_image(self, publisher, image):
        """Helper function to publish OpenCV debug images to ROS topics."""
        message = CompressedImage()
        _, encoded_data = cv2.imencode('.jpg', image)
        message.format = "jpeg"
        message.data = encoded_data.tobytes()
        publisher.publish(message)

    def get_vector_angle_in_radians(self, vector):
        """Calculates the slope angle of a vector in radians."""
        if ((vector[0][0] - vector[1][0]) == 0):  # Prevent division by zero
            theta = PI / 2
        else:
            slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
            theta = math.atan(slope)
        return theta

    def get_black_mask(self, image):
        """
        Builds a binary mask isolating the black lane boundary stripes using
        HSV + LAB color spaces instead of plain grayscale thresholding.

        Why this is more robust than grayscale alone:
        - Grayscale thresholding only looks at brightness. Under uneven lighting
          (sun glare, shadows from the buggy/track walls, etc.) parts of the white
          road can dip dark enough to false-positive, and parts of the black
          stripe can wash out bright enough to be missed.
        - HSV's V channel isolates brightness cleanly (no color mixed in like
          grayscale's weighted RGB sum does), so a V threshold is a cleaner
          "how dark is this" signal.
        - HSV's S channel lets us reject colored-but-dark pixels (colored
          shadows, tinted track walls, etc.) that aren't the true black stripe.
        - LAB's L* channel is a perceptually-uniform lightness measure and acts
          as a second, independent "is this actually dark" vote, computed from
          a different color transform than HSV. Requiring both to agree cuts
          down false positives further.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        h, s, v = cv2.split(hsv)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Dark AND desaturated (per HSV) AND dark (per LAB) => call it "black".
        hsv_mask = cv2.inRange(v, 0, HSV_VALUE_MAX) & cv2.inRange(s, 0, HSV_SATURATION_MAX)
        lab_mask = cv2.inRange(l_channel, 0, LAB_L_MAX)

        combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)

        # Morphological open (erode+dilate) removes small speckle noise;
        # close (dilate+erode) fills small gaps inside the stripe contours.
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.morph_kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, self.morph_kernel)

        return cleaned_mask

    def compute_vectors_from_image(self, image, thresh):
        """
        Analyzes the binary threshold image and extracts left and right lane edge vectors.
        This uses basic contour finding and coordinates calculations.

        You can optimize this algorithm or implement alternative methods here.
        """
        # Find contours around black edge stripes detected in the binary threshold image.
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        vectors = []
        for i in range(len(contours)):
            coordinates = contours[i][:, 0, :]

            # Get coordinates representing the boundaries of the contour
            min_y_value = np.min(coordinates[:, 1])
            max_y_value = np.max(coordinates[:, 1])

            min_y_coords = np.array(coordinates[coordinates[:, 1] == min_y_value])
            max_y_coords = np.array(coordinates[coordinates[:, 1] == max_y_value])

            min_y_coord = min_y_coords[0]
            max_y_coord = max_y_coords[0]

            # Calculate contour vector magnitude
            magnitude = np.linalg.norm(min_y_coord - max_y_coord)
            if (magnitude > VECTOR_MAGNITUDE_MINIMUM):
                # Calculate distance from the camera center at the bottom of the crop
                rover_point = [self.image_width / 2, self.lower_image_height]
                middle_point = (min_y_coord + max_y_coord) / 2
                distance = np.linalg.norm(middle_point - rover_point)

                # Correct point coordinates based on vector slope angle
                angle = self.get_vector_angle_in_radians([min_y_coord, max_y_coord])
                if angle > 0:
                    min_y_coord[0] = np.max(min_y_coords[:, 0])
                else:
                    max_y_coord[0] = np.max(max_y_coords[:, 0])

                # Store vectors with distance metadata for sorting
                vectors.append([list(min_y_coord), list(max_y_coord), distance])

            # Draw all detected raw vectors in blue on the debug image
            cv2.line(image, tuple(min_y_coord), tuple(max_y_coord), BLUE_COLOR, 2)

        return vectors, image

    def process_image_for_edge_vectors(self, image):
        """
        Applies HSV/LAB-based color thresholding and extracts lane vectors.

        OPTIMIZATION HINTS (still open for you to try):
        - Inverse Perspective Mapping (IPM) / Perspective Warp: Transforming the image into a
          "birds-eye view" before processing makes steering math linear and much easier to calculate.
        - Regions of Interest (ROI): Make sure to filter out the sky, horizon, or the buggy's own chassis
          to avoid spurious noise.
        - Sliding Windows / Polynomial Fitting: Instead of simple lines, you could fit a quadratic curve
          (y = Ax^2 + Bx + C) to handle bends, curves, and intersections more smoothly.
        - Tune HSV_VALUE_MAX / HSV_SATURATION_MAX / LAB_L_MAX at the top of this file if the mask is
          too noisy (lower HSV_VALUE_MAX) or missing faint stripes (raise HSV_VALUE_MAX slightly).
        """
        self.image_height, self.image_width, _ = image.shape
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = int(self.image_height - self.lower_image_height)

        # 1. Build a binary "black stripe" mask using HSV + LAB (replaces plain grayscale threshold).
        thresh = self.get_black_mask(image)

        # 2. Crop the image to focus on the lower section close to the buggy
        thresh_cropped = thresh[self.image_height - self.lower_image_height:]
        image_cropped = image[self.image_height - self.lower_image_height:].copy()

        # 3. Compute vectors from the binary image contours
        vectors, debug_img = self.compute_vectors_from_image(image_cropped, thresh_cropped)

        # 4. Sort vectors based on distance from buggy (we prioritize vectors closer to us)
        vectors = sorted(vectors, key=lambda x: x[2])

        # 5. Split vectors based on left/right halves of the image
        half_width = self.image_width / 2
        vectors_left = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) < half_width]
        vectors_right = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) >= half_width]

        final_vectors = []
        # Select the closest vector from each side (left/right)
        for side_vectors in [vectors_left, vectors_right]:
            if len(side_vectors) > 0:
                best_vector = side_vectors[0]
                # Draw the selected key lane vector in green on the debug image
                cv2.line(debug_img, tuple(best_vector[0]), tuple(best_vector[1]), GREEN_COLOR, 2)

                # Transform coordinates back to the original uncropped image space
                best_vector[0][1] += self.upper_image_height
                best_vector[1][1] += self.upper_image_height
                final_vectors.append(best_vector[:2])

        # Publish visual debugging images (viewable in tools like Foxglove)
        self.publish_debug_image(self.publisher_thresh_image, thresh_cropped)
        self.publish_debug_image(self.publisher_vector_image, debug_img)

        return final_vectors

    def camera_image_callback(self, message):
        """Processes incoming camera frames and publishes detected EdgeVectors."""
        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        vectors = self.process_image_for_edge_vectors(image)

        # Construct and publish the ROS 2 EdgeVectors message
        vectors_message = EdgeVectors()
        vectors_message.image_height = image.shape[0]
        vectors_message.image_width = image.shape[1]
        vectors_message.vector_count = 0

        # Vector 1 (usually representing Left boundary)
        if len(vectors) > 0:
            vectors_message.vector_1[0].x = float(vectors[0][0][0])
            vectors_message.vector_1[0].y = float(vectors[0][0][1])
            vectors_message.vector_1[1].x = float(vectors[0][1][0])
            vectors_message.vector_1[1].y = float(vectors[0][1][1])
            vectors_message.vector_count += 1

        # Vector 2 (usually representing Right boundary)
        if len(vectors) > 1:
            vectors_message.vector_2[0].x = float(vectors[1][0][0])
            vectors_message.vector_2[0].y = float(vectors[1][0][1])
            vectors_message.vector_2[1].x = float(vectors[1][1][0])
            vectors_message.vector_2[1].y = float(vectors[1][1][1])
            vectors_message.vector_count += 1

        self.publisher_edge_vectors.publish(vectors_message)


def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
