# save_crnn_to_onnx.py
import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(512), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU()
        )
        self.rnn1 = nn.LSTM(512, nh, bidirectional=True)
        self.rnn2 = nn.LSTM(nh * 2, nh, bidirectional=True)
        self.linear = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "height must be 1"
        conv = conv.squeeze(2).permute(2, 0, 1)  # [w, b, c]
        rnn_out, _ = self.rnn1(conv)
        rnn_out, _ = self.rnn2(rnn_out)
        output = self.linear(rnn_out)
        return output

# 경로
MODEL_DIR = Path(__file__).resolve().parent / "models"
crnn_pth_path = MODEL_DIR / "crnn.pth"
crnn_onnx_path = MODEL_DIR / "CRNN_VGG_BiLSTM_CTC.onnx"

# 모델 로드
alphabet = '0123456789가나다라마바사아자차카타파하'
nclass = len(alphabet) + 1
model = CRNN(32, 1, nclass, 256)

state_dict = torch.load(crnn_pth_path, map_location="cpu")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_k = k.replace("module.", "")
    new_state_dict[new_k] = v
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# ONNX로 변환
dummy_input = torch.randn(1, 1, 32, 100)
try:
    import onnx
    torch.onnx.export(
        model,
        dummy_input,
        crnn_onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {1: "batch"}},
        opset_version=11
    )
    print(f"✅ CRNN 모델이 ONNX로 저장되었습니다: {crnn_onnx_path}")
except ImportError:
    print("❌ onnx 패키지가 설치되어 있지 않습니다. 아래 명령어로 설치해주세요:\n\n    pip install onnx")
