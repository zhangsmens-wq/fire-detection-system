import os
import cv2


def detect_image(model, image_path, result_dir):
    # YOLO 预测
    results = model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )

    # 获取文件名
    file_name = os.path.basename(image_path)
    # 确保保存到 static/results 下
    save_path = os.path.join(result_dir, file_name)

    # 绘制结果并手动保存，这样路径最稳固
    for r in results:
        im_array = r.plot()
        cv2.imwrite(save_path, im_array)

    return file_name  # 返回文件名，方便前端拼接路径


def analyze_video_for_fire(model, video_path, result_dir, conf_threshold=0.5):
    # 此函数保留，供以后扩展使用
    return {"status": "success"}