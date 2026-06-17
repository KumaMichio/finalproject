from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("../runs/detect/runs/detect/yolo11m_vn_carla4/weights/last.pt")
    model.train(resume=True, workers=4)
