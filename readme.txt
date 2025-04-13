# 개발자 채용 인사이트 시스템 설정 및 실행 방법

# Python 3.11 설치 (윈도우의 경우 공식 웹사이트에서 설치)

# 가상환경 생성
python3.11 -m venv venv

# 가상환경 활성화
venv\Scripts\activate

pip install -r requirements.txt --no-cache-dir
# 1. 필요한 패키지 설치 - 주석은 앞으로 영어로 할것
set PYTHONUTF8=1
pip install -r requirements.txt

# 2. Redis 설치 (Celery 메시지 브로커로 사용)
# Windows:
# https://github.com/tporadowski/redis/releases 에서 Redis for Windows 다운로드 및 설치
# 또는 WSL을 통해 Linux 버전 설치
# Linux/Mac:
# sudo apt-get install redis-server (Ubuntu)
# brew install redis (Mac)

pip install --only-binary :all: mysqlclient
# 3. 환경 변수 설정 (윈도우)
set JOBKOREA_USERNAME=your_username
set JOBKOREA_PASSWORD=your_password

# 4. 데이터베이스 마이그레이션
python manage.py makemigrations
python manage.py migrate

# 5. 채용 정보 크롤링 수동 실행 (백업 옵션 포함)
# 기본 실행 (DB에만 저장)
python manage.py crawl_jobs --pages 3

# CSV 파일로 백업하기
python manage.py crawl_jobs --pages 3 --backup-format csv

# Excel 파일로 백업하기
python manage.py crawl_jobs --pages 3 --backup-format excel

# 모든 형식으로 백업하기
python manage.py crawl_jobs --pages 3 --backup-format both

# 6. Celery 작업자 및 스케줄러 실행
# 별도의 터미널에서 실행:

# Celery 워커 시작 (작업 처리)
celery -A jobproject worker --loglevel=info

# Celery Beat 시작 (작업 스케줄링)
celery -A jobproject beat --loglevel=info

# 또는 워커와 비트를 함께 실행:
celery -A jobproject worker --beat --loglevel=info

# 7. Django 서버 실행
python manage.py runserver

# 8. 웹사이트 접속
# - 메인 페이지: http://localhost:8000/
# - 채용 목록: http://localhost:8000/jobs/
# - 인사이트 페이지: http://localhost:8000/insights/
# - 관리자 페이지: http://localhost:8000/admin/ (태스크 스케줄 관리)

# 9. 백업 데이터 위치
# 백업 파일은 프로젝트 루트의 'backup_data' 폴더에 저장.
# 파일명 형식: job_data_YYYYMMDD_HHMMSS.csv 또는 job_data_YYYYMMDD_HHMMSS.xlsx

# 10. 예약된 크롤링 작업
# - 매일 새벽 2시: 3페이지 크롤링, CSV 형식으로 백업
# - 매주 일요일 새벽 3시: 10페이지 크롤링, CSV 및 Excel 형식으로 백업
# - 매월 1일 새벽 4시: 30일 이상 된 백업 파일 자동 정리

# 11. 관리자 페이지에서 스케줄 관리
# Django 관리자 페이지에 로그인한 후, "Periodic Tasks" 섹션에서 
# 예약된 태스크를 확인 및 수정.
# 관리자 계정 생성: python manage.py createsuperuser

# 12. 모델 학습 및 예측
<구현을 위한 다음 단계>
이 모델들을 실제로 통합하기 위해 다음과 같은 단계가 필요합니다:

모델 학습 페이지 추가:

관리자가 크롤링된 데이터를 사용해 모델을 학습할 수 있는 페이지 구현
학습 진행 상황 및 결과 표시
모델 예측 결과를 인사이트 페이지에 통합:

새로운 탭이나 섹션을 추가하여 ML/DL 예측 결과 표시
예측 결과를 시각화하는 차트 구현
정기적인 모델 재학습 자동화:

Celery 작업에 모델 재학습 태스크 추가
새로운 데이터가 충분히 쌓이면 자동으로 모델 업데이트