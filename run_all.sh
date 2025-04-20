#!/bin/bash
# 통합 실행 스크립트: Flask 백엔드 + React 프론트엔드
# .env에서 PORT 읽고 Flask + React 자동 실행

# -----------------------------------
# 1. 환경 변수 로딩 (.env)
# -----------------------------------
ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
  echo "[❌] .env 파일이 존재하지 않습니다. 예시 파일을 참고해 생성하세요: .env.example"
  exit 1
fi

# .env 로드
export $(grep -v '^#' "$ENV_FILE" | xargs)

# 기본값 설정
PORT=${PORT:-5050}
FRONT_DIR=${FRONT_DIR:-plate-web}

# -----------------------------------
# 2. Flask 백엔드 실행
# -----------------------------------
echo "🚀 Flask API 서버를 시작합니다 (PORT: $PORT)"
FLASK_CMD="python3 vision_api.py"
FLASK_ENV=production FLASK_APP=vision_api.py $FLASK_CMD &

FLASK_PID=$!
echo "🟢 Flask PID: $FLASK_PID"

# -----------------------------------
# 3. React 프론트엔드 실행
# -----------------------------------
echo "🌐 React 앱을 $FRONT_DIR 에서 실행합니다..."
cd "$FRONT_DIR" || exit 1

npm install
npm start
