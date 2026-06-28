from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\Users\Admin\finalproject\runs\detect\runs\detect\plate_chars_detector\weights\last.pt")
    model.train(resume=True, workers=2)
