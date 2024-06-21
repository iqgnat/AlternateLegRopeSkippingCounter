import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np


class BufferList:
    def __init__(self, buffer_time, default_value=0):
        self.buffer = [default_value for _ in range(buffer_time)]

    def push(self, value):
        self.buffer.pop(0)
        self.buffer.append(value)

    def max(self):
        return max(self.buffer)

    def min(self):
        buffer = [value for value in self.buffer if value]
        if buffer:
            return min(buffer)
        return 0


# Function to format time as mm:ss:ms
def format_time(milliseconds):
    seconds, ms = divmod(milliseconds, 1000)
    m, s = divmod(seconds, 60)
    return '{:02}:{:02}:{:03}'.format(int(m), int(s), int(ms))


# file
file_name = "example_01.mp4"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# the following code is to get the resolution of video, to help decide swap_misjudgment_pixel_buffer for leg swap
# cap = cv2.VideoCapture('jump_rope_02.mp4')
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(f"Video resolution: {width}x{height}")
# cap.release()

buffer_time = 80  # 80
swap_misjudgment_pixel_buffer = 50  # adjust pixels here based on your specific video resolution and leg lifting amplitude
center_y = BufferList(buffer_time)
center_y_up = BufferList(buffer_time)
center_y_down = BufferList(buffer_time)
center_y_pref_flip = BufferList(buffer_time)
center_y_flip = BufferList(buffer_time)
timer = 0  # display seconds
frames_per_second = 30
prev_leg_position = "left"
count = 0

# For webcam input:
cap = cv2.VideoCapture(file_name)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    file_name.replace(".mp4", "_output.mp4"),
    fourcc,
    frames_per_second,
    (int(cap.get(3)), int(cap.get(4))),
)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_height, image_width, _ = image.shape

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            landmarks = [(lm.x * image_width, lm.y * image_height) for lm in results.pose_landmarks.landmark]
            cx = int(np.mean([x[0] for x in landmarks]))
            cy = int(np.mean([x[1] for x in landmarks]))
            # Calculate relevant landmarks for legs
            left_heel = landmarks[29]  # left heel
            right_heel = landmarks[30]  # right heel

            # Calculate vertical positions
            left_y = left_heel[1]
            right_y = right_heel[1]

            # Detect leg swap or cross, to avoid false positive judgement
            if left_y < right_y - swap_misjudgment_pixel_buffer:
                leg_position = "left"

            if right_y < left_y - swap_misjudgment_pixel_buffer:
                leg_position = "right"

            # Track leg position changes
            if prev_leg_position != leg_position:
                leg_swapped = True
            else:
                leg_swapped = False

            prev_leg_position = leg_position

            # Count leg swaps
            if leg_swapped:
                count += 1
        else:
            cx = 0
            cy = 0
            cy_shoulder_hip = 0

        # print('count:' + str(count))

        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            image,
            "centroid",
            (cx - 25, cy - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        # Update the jump count
        cv2.putText(
            image,
            "count = " + str(count),
            (int(image_width * 0.3), int(image_height * 0.3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,  # 0.5
            (0, 255, 0),  # (0, 255, 0)
            3,  # 1
        )
        # Calculate and update the timer
        cv2.putText(
            image,
            "Time = " + format_time(timer * 1000 // frames_per_second),
            (int(image_width * 0.3), int(image_height * 0.25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,  # 0.5
            (255, 255, 255),  # (0, 255, 0)
            3,  # 1
        )
        timer += 1

        plt.clf()
        plt.plot(center_y.buffer, label="center_y")
        plt.plot(center_y_up.buffer, label="center_y_up")
        plt.plot(center_y_down.buffer, label="center_y_down")
        plt.plot(center_y_pref_flip.buffer, label="center_y_pref_flip")
        plt.plot(center_y_flip.buffer, label="center_y_flip")
        plt.legend(loc="upper right")
        # plt.pause(0.1)

        # display.
        cv2.imshow("MediaPipe Pose", image)
        out.write(image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
