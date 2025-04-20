# License Plate Recognition Web App

AI 기반 차량 번호판 인식 시스템  
(Flask API + YOLOv5 + EasyOCR + React 웹 연동)

---

## 📁 프로젝트 구조
licenseplate/
├── images/           # 테스트용 원본 이미지 저장 폴더
├── models/           # 학습된 모델 파일들 (YOLOv5, CRNN, CRAFT 등)
├── res_web/          # 인식 결과 이미지 저장 폴더
├── uploads/          # 업로드된 이미지 저장 위치
├── yolov5/           # YOLOv5 소스코드 디렉토리
├── plate-web/        # React 웹 프론트엔드 프로젝트
├── vision_server.py  # 번호판 인식 처리 함수 정의 (YOLO + EasyOCR)
├── vision_api.py     # Flask 기반 REST API 서버 실행 파일
├── requirements.txt  # Python 패키지 의존성 목록
└── .env.example      # 환경 변수 설정 예시 (.env 템플릿)
