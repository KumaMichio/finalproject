from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/yolo11m_vn.pt")
    model.train(
        data="data/visdrone_carla_vn2.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        workers=4,
        project="../runs/detect/runs/detect",
        name="yolo11m_vn_carla5",
    )
