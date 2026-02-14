
import cv2
import numpy as np
import imutils


class BallTracker:
    def __init__(self):
        # 定义小球的HSV颜色范围（这里以橙色/黄色小球为例）
        # 你需要根据你的小球颜色调整这些值
        self.color_ranges = [
            # 橙色小球范围（示例）
            ((10, 100, 100), (25, 255, 255)),  # 橙色
            # 绿色小球范围
            ((36, 50, 50), (86, 255, 255)),  # 绿色
            # 蓝色小球范围
            ((94, 80, 2), (126, 255, 255)),  # 蓝色
            # 红色小球范围（注意红色在HSV中有两个范围）
            ((0, 100, 100), (10, 255, 255)),  # 红色范围1y
            ((160, 100, 100), (180, 255, 255))  # 红色范围2
        ]

        # 选择当前使用的颜色范围（默认使用橙色）
        self.current_color_index = 0

        # 追踪参数
        self.min_radius = 10  # 最小半径
        self.max_radius = 200  # 最大半径

    def process_frame(self, frame):
        """
        处理单帧图像，检测并追踪小球
        """
        # 复制原始帧用于显示
        display_frame = frame.copy()

        # 1. 预处理
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 2. 创建颜色掩码
        mask = None
        color_lower, color_upper = self.color_ranges[self.current_color_index]

        if self.current_color_index == 3:  # 红色需要两个掩码合并
            mask1 = cv2.inRange(hsv, color_lower, color_upper)
            color_lower2, color_upper2 = self.color_ranges[4]
            mask2 = cv2.inRange(hsv, color_lower2, color_upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, color_lower, color_upper)

        # 3. 形态学操作（去除噪声）
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 4. 查找轮廓
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        center = None
        radius = 0

        # 5. 如果找到轮廓
        if len(contours) > 0:
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)

            # 获取最小外接圆
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

            # 计算轮廓的矩
            M = cv2.moments(largest_contour)

            if M["m00"] > 0 and self.min_radius < radius < self.max_radius:
                # 计算中心点
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # 绘制圆形和中心点
                cv2.circle(display_frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(display_frame, center, 5, (0, 0, 255), -1)

                # 显示坐标和半径
                info_text = f"Center: ({center[0]}, {center[1]}), Radius: {int(radius)}"
                cv2.putText(display_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 6. 显示处理过程中的图像
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        processed_display = np.hstack([display_frame, mask_display])

        # 7. 添加控制说明
        instructions = [
            "Instructions:",
            "Q - Quit",
            "C - Change color",
            "+/- - Adjust min radius",
            "[ ] - Adjust max radius",
            "R - Reset tracking"
        ]

        y_offset = 60
        for instruction in instructions:
            cv2.putText(processed_display, instruction, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        return processed_display, center, radius

    def change_color(self):
        """切换到下一个颜色范围"""
        self.current_color_index = (self.current_color_index + 1) % 3
        color_names = ["Orange", "Green", "Blue", "Red"]
        print(f"Changed to {color_names[self.current_color_index]} color tracking")


def main():
    # 创建追踪器
    tracker = BallTracker()

    # 打开摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(0)

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Ball Tracking Program Started")
    print("Press 'q' to quit")
    print("Press 'c' to change color range")

    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # 处理帧
        processed_frame, center, radius = tracker.process_frame(frame)

        # 显示结果
        cv2.imshow("Ball Tracker - Press Q to quit", processed_frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # 退出
            break
        elif key == ord('c'):  # 切换颜色
            tracker.change_color()
        elif key == ord('+'):  # 增加最小半径
            tracker.min_radius += 5
            print(f"Min radius: {tracker.min_radius}")
        elif key == ord('-'):  # 减小最小半径
            tracker.min_radius = max(5, tracker.min_radius - 5)
            print(f"Min radius: {tracker.min_radius}")
        elif key == ord(']'):  # 增加最大半径
            tracker.max_radius += 5
            print(f"Max radius: {tracker.max_radius}")
        elif key == ord('['):  # 减小最大半径
            tracker.max_radius = max(tracker.min_radius + 10, tracker.max_radius - 5)
            print(f"Max radius: {tracker.max_radius}")
        elif key == ord('r'):  # 重置
            tracker.min_radius = 10
            tracker.max_radius = 200
            print("Parameters reset")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")


if __name__ == "__main__":
    main()