import cv2
import numpy as np

def compute_integral_image(gray):
    """
    计算灰度图的积分图
    """
    integral = cv2.integral(gray)
    return integral

def haar_feature_upper_black_lower_white(integral, x, y, w, h):
    """
    计算一个上黑下白的Haar特征值
    注意：由于 OpenCV 的积分图比原图大一，所以x和y需要加1
    """

    half_h = h // 2

    A = integral[y, x]
    B = integral[y, x + w]
    C = integral[y + half_h, x]
    D = integral[y + half_h, x + w]
    upper_sum = D - B - C + A
    A = integral[y + half_h, x]
    B = integral[y + half_h, x + w]
    C = integral[y + h, x]
    D = integral[y + h, x + w]
    lower_sum = D - B - C + A

    feature = upper_sum - lower_sum
    return feature

def sliding_window_detection(integral, window_size, step_size, threshold):
    """
    在积分图上使用滑动窗口检测Haar特征
    """
    detections = []
    h_integral, w_integral = integral.shape
    h_window, w_window = window_size

    for y in range(0, h_integral - h_window):
        for x in range(0, w_integral - w_window):
            feature = haar_feature_upper_black_lower_white(integral, x, y, w_window, h_window)
            if feature > threshold:
                detections.append((x, y, w_window, h_window, feature))

        y += step_size

    return detections

def non_max_suppression(detections, overlap_thresh):
    """
    非极大值抑制，消除重叠的检测框
    """
    if len(detections) == 0:
        return []

    boxes = np.array([[x, y, x + w, y + h, score] for (x, y, w, h, score) in detections])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(detections[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        overlap = intersection / (areas[i] + areas[order[1:]] - intersection)
        inds = np.where(overlap <= overlap_thresh)[0]
        order = order[inds + 1]

    return keep

def pyramid(image, scale=1.5, min_size=(24, 24)):
    """
    生成图像金字塔
    """
    yield image

    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))
        if w < min_size[0] or h < min_size[1]:
            break
        yield image

def main():

    image_path = '/Users/a1/Documents/Image/lenna.jpg'  
    window_size = (128, 128)   
    step_size = 16            
    threshold = 100000         
    overlap_thresh = 0.1    


    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像 {image_path}")
        return

    gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections_all = []

    for scale_idx, resized in enumerate(pyramid(gray_orig, scale=1.5, min_size=window_size)):
        scale = gray_orig.shape[1] / float(resized.shape[1])
        integral = compute_integral_image(resized)

        detections = sliding_window_detection(integral, window_size, step_size, threshold)
        print(f"尺度 {scale_idx +1 } 检测到 {len(detections)} 个候选框")

        for (x, y, w, h, score) in detections:
            detections_all.append((int(x * scale), int(y * scale), int(w * scale), int(h * scale), score))

    print(f"初步检测到 {len(detections_all)} 个候选框")

    final_detections = non_max_suppression(detections_all, overlap_thresh)
    print(f"经过非极大值抑制后检测到 {len(final_detections)} 个人脸")
    
    for (x, y, w, h, score) in final_detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
