from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("../runs/runs/detect/runs/detect/yolo11m_vn_carla6-3/weights/last.pt")
    model.train(resume=True, workers=4)
