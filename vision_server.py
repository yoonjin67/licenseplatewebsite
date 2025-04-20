# vision_server.py
import sys, os, re, cv2, torch
import numpy as np
from pathlib import Path
import easyocr
from PIL import ImageFont, ImageDraw, Image

ROOT = Path(__file__).resolve().parent
YOLOV5_DIR = ROOT / 'yolov5'
MODEL_PATH = ROOT / 'best.pt'
FONT_PATH = ROOT / 'NanumGothic-Bold.ttf'
RES_DIR = ROOT / 'res_web'
RES_DIR.mkdir(exist_ok=True)
sys.path.append(str(YOLOV5_DIR))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox

reader = easyocr.Reader(['ko', 'en'], gpu=False)
model = DetectMultiBackend(MODEL_PATH, device='cpu')

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

def draw_text_with_pil(image, text, position, font_path=FONT_PATH, font_size=32):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(str(font_path), font_size)
    draw.text(position, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def process_image(image_path):
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise ValueError(f"이미지를 열 수 없습니다: {image_path}")

    img = letterbox(img0, new_shape=640)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)

    pred = model(img)
    pred = non_max_suppression(pred, 0.3, 0.45)

    results = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, _, _ in det:
                x1, y1, x2, y2 = map(int, xyxy)
                crop = img0[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                ocr_results = reader.readtext(gray, detail=0)
                raw_text = ''.join(ocr_results)
                text = re.sub(r'[^0-9가-힣]', '', raw_text)
                match = re.match(r'(\d{2,3})([가-힣])(\d{4})', text)
                result = f"{match.group(1)}{match.group(2)} {match.group(3)}" if match else text or "NoText"
                results.append(result)
                img0 = draw_text_with_pil(img0, result, (x1, y1 - 40))
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = '_'.join(results) if results else 'unlabeled'
    save_path = RES_DIR / f"{Path(image_path).stem}_{label}.jpg"
    cv2.imwrite(str(save_path), img0)

    return str(save_path), label

