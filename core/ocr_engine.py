def read_plate_yolo26(plate_img, model_s3):
    results = model_s3.predict(plate_img, conf=0.4, imgsz=320, verbose=False)[0]
    
    chars = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model_s3.names[int(box.cls[0])] 
        chars.append({'char': label, 'x': (x1+x2)/2, 'y': (y1+y2)/2, 'h': y2-y1})
    
    if not chars: return ""

    chars.sort(key=lambda c: c['y'])
    y_threshold = chars[0]['y'] + chars[0]['h'] * 0.6 
    
    line1 = [c for c in chars if c['y'] < y_threshold]
    line2 = [c for c in chars if c['y'] >= y_threshold]
    
    line1.sort(key=lambda c: c['x'])
    line2.sort(key=lambda c: c['x'])
    
    str1 = "".join([c['char'] for c in line1])
    str2 = "".join([c['char'] for c in line2])
    
    return f"{str1}-{str2}" if str2 else str1