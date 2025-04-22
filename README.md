# License Plate Recognition Web App

AI 기반 차량 번호판 인식 시스템  
(Flask API + YOLOv5 + EasyOCR + React 웹 연동)

---

## 📁 프로젝트 구조
licenseplate/
├── images/           # 테스트용 원본 이미지 저장 폴더
├── models/           # 학습된 모델 파일들 (YOLOv5, CRNN, CRAFT 등)
├── res\_web/          # 인식 결과 이미지 저장 폴더
├── uploads/          # 업로드된 이미지 저장 위치
├── yolov5/           # YOLOv5 소스코드 디렉토리
├── plate-web/        # React 웹 프론트엔드 프로젝트
├── vision\_server.py  # 번호판 인식 처리 함수 정의 (YOLO + EasyOCR)
├── vision\_api.py     # Flask 기반 REST API 서버 실행 파일
├── requirements.txt  # Python 패키지 의존성 목록
└── .env.example      # 환경 변수 설정 예시 (.env 템플릿)


❗ Netlify에서는 AI 기능이 작동하지 않습니다
이 프로젝트는 차량 번호판 인식(AI OCR)을 위한 웹앱입니다.
다만 Netlify에서는 AI 기능(YOLOv5 + EasyOCR)이 동작하지 않습니다. 이유는 다음과 같습니다:

Netlify는 정적 파일만 서빙하며, Python 기반의 AI 서버(flask 등)를 지원하지 않습니다.

YOLOv5, EasyOCR 등 AI 모델은 브라우저에서 직접 실행 불가능하고, 별도의 백엔드 서버가 필요합니다.

현재의 AI 기능은 Flask 서버에서 실행되며, Netlify는 이 서버와 통신할 수 없습니다.

✅ 해결 방법
AI 서버를 별도로 배포(예: AWS, Render, Railway, PythonAnywhere 등)

또는 Netlify Functions + 경량 모델 (제한적 지원)

👉 따라서 실제 AI OCR 기능을 사용하려면 Flask API 서버와 함께 사용해야 합니다.


