#!/bin/bash

# 변환할 CoreML 모델 파일 이름
MLMODEL_FILE="best.mlmodel"

# 출력될 Huawei 실행 파일 이름
HWX_FILE="${MLMODEL_FILE%.mlmodel}.hwx"

# 출력될 ANE 모델 파일 이름
ANEC_FILE="${MLMODEL_FILE%.mlmodel}.anec"

# tohwx 실행 파일 경로 (anecc 저장소 내의 tohwx 디렉토리 기준)
TOHWX_PATH="./anecc/tohwx/tohwx"

# anecc 실행 파일 경로 (anecc 저장소 루트 기준)
ANECC_PATH="./anecc/anecc"

# tohwx를 사용하여 .mlmodel -> .hwx 변환 (macOS에서 실행 필요)
echo "macOS에서 ${MLMODEL_FILE}을 ${HWX_FILE}으로 변환합니다..."
if [ -f "$TOHWX_PATH" ]; then
  "$TOHWX_PATH" "$MLMODEL_FILE"
  if [ -f "$HWX_FILE" ]; then
    echo "${HWX_FILE} 파일 생성 완료."
  else
    echo "오류: ${HWX_FILE} 파일 생성 실패. tohwx 실행을 확인하세요."
    exit 1
  fi
else
  echo "오류: tohwx 실행 파일(${TOHWX_PATH})을 찾을 수 없습니다. anecc 저장소 경로를 확인하세요."
  exit 1
fi

echo ""

# anecc를 사용하여 .hwx -> .anec 변환 (플랫폼 독립적)
echo "${HWX_FILE}을 ${ANEC_FILE}으로 변환합니다..."
if [ -f "$ANECC_PATH" ]; then
  "$ANECC_PATH" "$HWX_FILE" -o "$ANEC_FILE"
  if [ -f "$ANEC_FILE" ]; then
    echo "${ANEC_FILE} 파일 생성 완료."
    echo "변환 완료: ${MLMODEL_FILE} -> ${HWX_FILE} -> ${ANEC_FILE}"
  else
    echo "오류: ${ANEC_FILE} 파일 생성 실패. anecc 실행을 확인하세요."
    exit 1
  fi
else
  echo "오류: anecc 실행 파일(${ANECC_PATH})을 찾을 수 없습니다. anecc 저장소 경로를 확인하세요."
  exit 1
fi

exit 0
