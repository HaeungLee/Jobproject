import os
from celery import Celery
from celery.schedules import crontab

# Django 설정 모듈을 기본값으로 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jobproject.settings')

app = Celery('jobproject')

# 설정 모듈에서 'CELERY'로 시작하는 설정값을 읽어옵니다.
app.config_from_object('django.conf:settings', namespace='CELERY')

# 등록된 Django 앱에서 tasks.py 모듈을 자동으로 검색합니다.
app.autodiscover_tasks()

# 주기적인 태스크 정의
app.conf.beat_schedule = {
    'crawl-job-listings-daily': {
        'task': 'jobapp.tasks.crawl_job_listings',
        'schedule': crontab(hour=2, minute=0),  # 매일 새벽 2시에 실행
        'args': (3, 'csv'),  # 3 페이지, CSV 형식으로 백업
    },
    'crawl-job-listings-weekly': {
        'task': 'jobapp.tasks.crawl_job_listings',
        'schedule': crontab(day_of_week=0, hour=3, minute=0),  # 매주 일요일 새벽 3시에 실행
        'args': (10, 'both'),  # 10 페이지, CSV와 Excel 모두 백업
    },
    'cleanup-old-backups-monthly': {
        'task': 'jobapp.tasks.cleanup_old_backups',
        'schedule': crontab(day_of_month=1, hour=4, minute=0),  # 매월 1일 새벽 4시에 실행
        'args': (30,),  # 30일 이상 된 백업 파일 삭제
    },
}

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')