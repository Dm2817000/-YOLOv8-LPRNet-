from ultralytics import YOLO

model=YOLO('yolov8n.pt')
model.train(data="data-bvn.yaml",workers=0,epochs=50,batch=16)