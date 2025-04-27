# detect_plate4.py (번호판 전체 OCR 기반 - EasyOCR + 한글 표시 지원)
import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import csv
import re
import easyocr
from PIL import ImageFont, ImageDraw, Image

# ---------------------------------------------
# 경로 설정
# ---------------------------------------------
ROOT = Path(__file__).resolve().parent
YOLOV5_DIR = ROOT / 'yolov5'
MODEL_DIR = ROOT / 'models'
IMAGES_DIR = ROOT / 'images'
RES_DIR = ROOT / 'res9'
RES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(YOLOV5_DIR))

# ---------------------------------------------
# YOLOv5 불러오기
# ---------------------------------------------
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox

# scale_coords 직접 정의 (YOLOv5 최신버전 대응)
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords
scale_coords_fn = scale_coords

# ---------------------------------------------
# EasyOCR 설정
# ---------------------------------------------
reader = easyocr.Reader(['ko', 'en'], gpu=False) # 필요에 따라 True로 변경 가능

# ---------------------------------------------
# OpenCV 이미지에 한글 텍스트를 넣는 함수
# ---------------------------------------------
def draw_text_with_pil(image, text, position, font_path='NanumGothic-Bold.ttf', font_size=32):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# ---------------------------------------------
# YOLOv5 차량 번호판 탐지 및 문자 인식 수행
# ---------------------------------------------
if __name__ == '__main__':
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model = DetectMultiBackend(ROOT / 'best.pt', device=device)
    model.to(device) # 모델을 해당 장치로 이동

    with open(RES_DIR / 'recognition_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Detected Text'])

    img_files = sorted(IMAGES_DIR.glob("*.jpg"))
    for img_path in img_files:
        img0 = cv2.imread(str(img_path))
        img = letterbox(img0, new_shape=640)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0).to(device) # 이미지 텐서를 해당 장치로 이동

        pred = model(img)
        pred = non_max_suppression(pred, 0.3, 0.45)

        results = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords_fn(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    crop = img0[y1:y2, x1:x2]
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 20, 7)
                    ocr_results = reader.readtext(gray, detail=0)
                    raw_text = ''.join(ocr_results)
                    text = re.sub(r'[^0-9가-힣]', '', raw_text)
                    match = re.match(r'(\d{2,3})([가-힣]{1})(\d{4})', text)
                    result = f"{match.group(1)}{match.group(2)} {match.group(3)}" if match else text or "NoText"

                    results.append(result)
                    cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    img0 = draw_text_with_pil(img0, result, (x1, y1 - 40))

        label_text = '_'.join(results) if results else 'unlabeled'
        save_path = RES_DIR / f"{img_path.stem}_{label_text}.jpg"
        cv2.imwrite(str(save_path), img0)
        with open(RES_DIR / 'recognition_results.csv', 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([img_path.name, label_text])
