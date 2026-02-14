"""
OpenCV常用函数综合示例
包含：图像读写、颜色空间转换、几何变换、滤波、阈值、形态学、
边缘检测、轮廓、直方图、特征检测、模板匹配、绘图、视频处理等
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================== 辅助显示函数 ==================
def show_image(img, title='image', wait=True):
    """显示图像（使用OpenCV）"""
    cv2.imshow(title, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def show_images(images, titles, rows=1, cols=None, figsize=(12,8)):
    """使用matplotlib显示多张图像"""
    if cols is None:
        cols = len(images)
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i+1)
        # 如果是彩色图像，需要将BGR转为RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ================== 主程序 ==================
def main():
    # ------------------ 1. 读取图像 ------------------
    # 使用内置lena图像（如果没有，可生成或下载）
    # 尝试从文件读取，若失败则创建一个简单图像
    img_path = 'lena.jpg'  # 请确保存在该图片，或替换为自己的图片
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError
    except:
        print("未找到图片，生成一个测试图像（彩色渐变+形状）")
        # 创建一个512x512的彩色图像
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        # 绘制渐变
        for i in range(512):
            img[:, i] = [i//2, i//2, i//2]  # BGR渐变
        # 绘制一些形状
        cv2.rectangle(img, (100,100), (200,200), (0,255,0), -1)  # 绿色矩形
        cv2.circle(img, (300,300), 80, (255,0,0), -1)            # 蓝色圆
        cv2.line(img, (400,400), (500,500), (0,0,255), 5)        # 红色线

    # 图像基本信息
    h, w, c = img.shape
    print(f"原始图像尺寸: {w}x{h}, 通道数: {c}")

    # 显示原图
    show_image(img, 'Original', wait=False)

    # ------------------ 2. 颜色空间转换 ------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    show_images([img, gray, hsv, yuv],
                ['BGR', 'GRAY', 'HSV', 'YUV'], rows=2, cols=2)

    # ------------------ 3. 几何变换 ------------------
    # 缩放
    resized = cv2.resize(img, (256, 256))  # 固定尺寸
    resized_ratio = cv2.resize(img, None, fx=0.5, fy=0.5)  # 比例缩放

    # 旋转
    center = (w//2, h//2)
    M_rotate = cv2.getRotationMatrix2D(center, 45, 1.0)  # 旋转45度
    rotated = cv2.warpAffine(img, M_rotate, (w, h))

    # 裁剪
    cropped = img[100:300, 100:300]

    # 翻转
    flip_h = cv2.flip(img, 1)  # 水平翻转
    flip_v = cv2.flip(img, 0)  # 垂直翻转
    flip_hv = cv2.flip(img, -1) # 同时翻转

    show_images([resized, rotated, cropped, flip_h],
                ['Resized', 'Rotated', 'Cropped', 'Flip Horizontal'], rows=2, cols=2)

    # ------------------ 4. 图像滤波 ------------------
    # 高斯模糊
    gaussian = cv2.GaussianBlur(img, (5,5), 0)
    # 中值滤波
    median = cv2.medianBlur(img, 5)
    # 均值滤波
    blur = cv2.blur(img, (5,5))
    # 双边滤波
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    show_images([img, gaussian, median, blur, bilateral],
                ['Original', 'Gaussian', 'Median', 'Blur', 'Bilateral'], rows=2, cols=3)

    # ------------------ 5. 阈值处理 ------------------
    gray_img = gray.copy()
    # 简单二值化
    ret, thresh_binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    # 反二值化
    ret, thresh_binary_inv = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    # Otsu阈值
    ret, thresh_otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 自适应阈值
    thresh_adapt = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

    show_images([gray_img, thresh_binary, thresh_binary_inv, thresh_otsu, thresh_adapt],
                ['Gray', 'Binary', 'Binary Inv', 'Otsu', 'Adaptive'], rows=2, cols=3)

    # ------------------ 6. 形态学操作 ------------------
    # 创建结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # 腐蚀
    eroded = cv2.erode(thresh_otsu, kernel, iterations=1)
    # 膨胀
    dilated = cv2.dilate(thresh_otsu, kernel, iterations=1)
    # 开运算
    opened = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)
    # 闭运算
    closed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
    # 形态学梯度
    gradient = cv2.morphologyEx(thresh_otsu, cv2.MORPH_GRADIENT, kernel)

    show_images([thresh_otsu, eroded, dilated, opened, closed, gradient],
                ['Otsu', 'Eroded', 'Dilated', 'Opened', 'Closed', 'Gradient'], rows=2, cols=3)

    # ------------------ 7. 边缘检测 ------------------
    # Canny
    edges_canny = cv2.Canny(gray_img, 50, 150)
    # Sobel
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)  # 计算梯度幅值
    sobel = np.uint8(np.clip(sobel, 0, 255))
    # Laplacian
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    laplacian = np.uint8(np.clip(laplacian, 0, 255))

    show_images([gray_img, edges_canny, sobel, laplacian],
                ['Gray', 'Canny', 'Sobel', 'Laplacian'], rows=2, cols=2)

    # ------------------ 8. 轮廓检测 ------------------
    # 使用Canny结果找轮廓
    contours, hierarchy = cv2.findContours(edges_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0,255,0), 2)  # 绿色轮廓

    # 计算轮廓特征
    if contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        print(f"第一个轮廓面积: {area}, 周长: {perimeter}")

        # 外接矩形
        x,y,w_cnt,h_cnt = cv2.boundingRect(cnt)
        cv2.rectangle(img_contours, (x,y), (x+w_cnt, y+h_cnt), (255,0,0), 2)  # 蓝色矩形

        # 最小外接矩形
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_contours, [box], 0, (0,0,255), 2)  # 红色

    show_image(img_contours, 'Contours')

    # ------------------ 9. 直方图 ------------------
    # 计算灰度直方图
    hist = cv2.calcHist([gray_img], [0], None, [256], [0,256])
    # 绘制直方图（使用matplotlib）
    plt.figure()
    plt.plot(hist)
    plt.title('Grayscale Histogram')
    plt.xlim([0,256])
    plt.show()

    # 直方图均衡化
    equ = cv2.equalizeHist(gray_img)
    show_images([gray_img, equ], ['Original Gray', 'Equalized'], rows=1, cols=2)

    # CLAHE（限制对比度自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray_img)
    show_images([gray_img, equ, clahe_img],
                ['Original', 'Equalized', 'CLAHE'], rows=1, cols=3)

    # ------------------ 10. 特征检测 ------------------
    # 使用SIFT (需要opencv-contrib-python)
    try:
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray_img, None)
        img_sift = cv2.drawKeypoints(gray_img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        show_image(img_sift, 'SIFT Keypoints')
    except AttributeError:
        print("SIFT不可用，尝试ORB...")
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(gray_img, None)
        img_orb = cv2.drawKeypoints(gray_img, kp, None, color=(0,255,0))
        show_image(img_orb, 'ORB Keypoints')

    # ------------------ 11. 模板匹配 ------------------
    # 创建模板（从原图中裁剪一小块）
    template = gray_img[100:200, 100:200]  # 假设区域
    w_t, h_t = template.shape[::-1]
    # 匹配
    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w_t, top_left[1] + h_t)
    # 绘制结果
    img_match = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_match, top_left, bottom_right, (0,255,0), 2)
    cv2.putText(img_match, f"Match: {max_val:.2f}", (top_left[0], top_left[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    show_image(img_match, 'Template Match')

    # ------------------ 12. 绘图函数 ------------------
    canvas = np.zeros((512,512,3), dtype=np.uint8)
    # 画线
    cv2.line(canvas, (0,0), (511,511), (255,0,0), 3)
    # 画矩形
    cv2.rectangle(canvas, (50,50), (200,200), (0,255,0), 2)
    # 画圆
    cv2.circle(canvas, (300,300), 50, (0,0,255), -1)  # 填充圆
    # 画椭圆
    cv2.ellipse(canvas, (400,400), (100,50), 0, 0, 180, (255,255,0), -1)
    # 多边形
    pts = np.array([[10,5], [50,30], [30,60]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(canvas, [pts], True, (255,0,255), 3)
    # 添加文本
    cv2.putText(canvas, 'OpenCV', (150,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    show_image(canvas, 'Drawing')

    # ------------------ 13. 图像金字塔 ------------------
    # 高斯金字塔
    lower = cv2.pyrDown(img)
    higher = cv2.pyrUp(lower)
    # 拉普拉斯金字塔
    laplacian_pyramid = cv2.subtract(img, higher)  # 近似拉普拉斯
    show_images([img, lower, higher, laplacian_pyramid],
                ['Original', 'PyrDown', 'PyrUp', 'Laplacian'], rows=2, cols=2)

    # ------------------ 14. 仿射变换与透视变换 ------------------
    # 仿射变换
    rows, cols, _ = img.shape
    pts1 = np.float32([[50,50], [200,50], [50,200]])
    pts2 = np.float32([[10,100], [200,50], [100,250]])
    M_affine = cv2.getAffineTransform(pts1, pts2)
    affine_dst = cv2.warpAffine(img, M_affine, (cols, rows))

    # 透视变换
    pts1_p = np.float32([[56,65], [368,52], [28,387], [389,390]])
    pts2_p = np.float32([[0,0], [300,0], [0,300], [300,300]])
    M_persp = cv2.getPerspectiveTransform(pts1_p, pts2_p)
    persp_dst = cv2.warpPerspective(img, M_persp, (300,300))

    show_images([img, affine_dst, persp_dst],
                ['Original', 'Affine', 'Perspective'], rows=1, cols=3)

    # ------------------ 15. 视频处理（框架） ------------------
    # 演示如何读取视频并逐帧处理（这里不实际运行，仅展示代码结构）
    def video_demo():
        cap = cv2.VideoCapture(0)  # 打开摄像头，0为默认摄像头
        # 设置视频编码和输出
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 对每一帧进行处理（例如转为灰度）
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 可以在这里添加更多处理
            cv2.imshow('Video', gray_frame)
            # 写入视频
            out.write(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR))  # 需要BGR
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # 提示：若需运行视频处理，取消下面注释
    # video_demo()

    # ------------------ 16. 保存图像 ------------------
    cv2.imwrite('output_image.jpg', img_contours)
    print("处理完成，图像已保存。")

if __name__ == "__main__":
    main()