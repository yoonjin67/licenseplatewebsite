import sys, os, re, cv2, torch
import unicodedata
import numpy as np
import math
from pathlib import Path
import easyocr
from PIL import ImageFont, ImageDraw, Image
from itertools import product

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

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
model = DetectMultiBackend(MODEL_PATH, device=device)
model.to(device)
reader = easyocr.Reader(['ko', 'en'], gpu=device)


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

def get_chosung(char):
    if '가' <= char <= '힣':
        offset = ord(char) - ord('가')
        return offset // (21 * 28)
    return -1

def get_jungsung(char):
    if '가' <= char <= '힣':
        offset = ord(char) - ord('가')
        return (offset // 28) % 21
    return -1

def get_jongsung(char):
    if '가' <= char <= '힣':
        offset = ord(char) - ord('가')
        return offset % 28
    return -1

def combine_hangul(chosung, jungsung, jongsung):
    return chr(0xAC00 + (chosung * 21 + jungsung) * 28 + jongsung)

def is_valid_plate_format(text):
    match = re.match(r'^(\d{2,3})([가-힣])(\d{4})$', text)
    if match:
        return re.match(r'^\d+$', match.group(3)) is not None
    return False

def process_image(image_path):
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise ValueError(f"이미지를 열 수 없습니다: {image_path}")

    img = letterbox(img0, new_shape=640)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0).to(device)

    pred = model(img)
    pred = non_max_suppression(pred, 0.3, 0.45)

    results = []
    final_result = "NoText"
    best_coords = None

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            x1_base, y1_base, x2_base, y2_base = map(int, det[0][:4])

            pad = 1
            for dx1 in range(-pad, pad + 1):
                for dy1 in range(-pad, pad + 1):
                    for dx2 in range(-pad, pad + 1):
                        for dy2 in range(-pad, pad + 1):
                            x1_pad, y1_pad, x2_pad, y2_pad = x1_base + dx1, y1_base + dy1, x2_base + dx2, y2_base + dy2

                            if x1_pad < x2_pad and y1_pad < y2_pad:
                                crop = img0[y1_pad:y2_pad, x1_pad:x2_pad]
                                try:
                                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                except:
                                    continue
                                ocr_results = reader.readtext(gray, detail=0)
                                raw_text = ''.join(ocr_results)
                                text = re.sub(r'[^0-9가-힣]', '', raw_text)

                                if is_valid_plate_format(text):
                                    match = re.match(r'^(\d{2,3})([가-힣])(\d{4})$', text)
                                    if match:
                                        result_list = list(text)
                                        valid_mid_chars = "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주바사아자외하허호배"
                                        best_restore_idx = -1
                                        best_restore = ''
                                        best_dist = 10000

                                        for idx, char in enumerate(result_list):
                                            if unicodedata.category(char) != 'Nd':
                                                current_chosung = get_chosung(char)
                                                current_jungsung = get_jungsung(char)
                                                current_jongsung = get_jongsung(char)

                                                for v in valid_mid_chars:
                                                    valid_chosung = get_chosung(v)
                                                    valid_jungsung = get_jungsung(v)
                                                    valid_jongsung = get_jongsung(v)

                                                    if current_chosung == valid_chosung:
                                                        dist = abs(current_jungsung - valid_jungsung) + abs(current_jongsung - valid_jongsung)
                                                        if dist < best_dist:
                                                            best_dist = dist
                                                            best_restore = v
                                                            best_restore_idx = idx

                                        if best_restore_idx != -1:
                                            result_list[best_restore_idx] = best_restore

                                        corrected_result = "".join(result_list)
                                        print(f"Pad: ({dx1}, {dy1}, {dx2}, {dy2}), OCR: {corrected_result}")
                                        final_result = f"{corrected_result[:len(match.group(1)) + 1]} {corrected_result[len(match.group(1)) + 1:]}"
                                        best_coords = (x1_pad, y1_pad, x2_pad, y2_pad)
                                        break
                            if final_result != "NoText":
                                break
                        if final_result != "NoText":
                            break
                    if final_result != "NoText":
                        break
                if final_result != "NoText":
                    break
            if final_result != "NoText":
                break

    if final_result != "NoText" and best_coords:
        results.append(final_result)
        img0 = draw_text_with_pil(img0, final_result, (best_coords[0], best_coords[1] - 40))
        cv2.rectangle(img0, (best_coords[0], best_coords[1]), (best_coords[2], best_coords[3]), (0, 255, 0), 2)

    label = '_'.join(results) if results else 'unlabeled'
    save_path = RES_DIR / f"{Path(image_path).stem}_{label}.jpg"
    cv2.imwrite(str(save_path), img0)

    return str(save_path), label
