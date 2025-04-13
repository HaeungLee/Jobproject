import logging
from celery import shared_task
from django.core.management import call_command
from django.conf import settings
from datetime import datetime

logger = logging.getLogger(__name__)

@shared_task(name='jobapp.tasks.crawl_job_listings')
def crawl_job_listings(pages=3, backup_format='csv'):
    """
    잡코리아 채용 공고를 크롤링하는 Celery 태스크
    
    Args:
        pages (int): 크롤링할 페이지 수
        backup_format (str): 백업 파일 형식 ('csv', 'excel', 'both')
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"[{timestamp}] 예약된 크롤링 작업 시작: {pages} 페이지, 백업 형식 '{backup_format}'")
    
    try:
        # Django 관리 명령 실행 (crawl_jobs.py)
        call_command('crawl_jobs', pages=pages, backup_format=backup_format)
        
        logger.info(f"[{timestamp}] 예약된 크롤링 작업 완료")
        return True
    except Exception as e:
        logger.error(f"[{timestamp}] 크롤링 작업 실패: {str(e)}")
        return False

@shared_task(name='jobapp.tasks.cleanup_old_backups')
def cleanup_old_backups(days=30):
    """
    오래된 백업 파일을 정리하는 태스크
    
    Args:
        days (int): 이 일수보다 오래된 파일을 삭제합니다.
    """
    import os
    import glob
    import time
    from datetime import datetime, timedelta
    
    backup_dir = os.path.join(settings.BASE_DIR, 'backup_data')
    if not os.path.exists(backup_dir):
        logger.info("백업 디렉토리가 존재하지 않습니다.")
        return
    
    # 현재 시간 기준으로 days일 이전 시간 계산
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    
    # 모든 백업 파일 찾기
    backup_files = glob.glob(os.path.join(backup_dir, 'job_data_*.csv'))
    backup_files.extend(glob.glob(os.path.join(backup_dir, 'job_data_*.xlsx')))
    
    removed_count = 0
    for file_path in backup_files:
        file_mod_time = os.path.getmtime(file_path)
        if file_mod_time < cutoff_time:
            try:
                os.remove(file_path)
                removed_count += 1
                logger.info(f"오래된 백업 파일 삭제: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"파일 삭제 실패: {os.path.basename(file_path)} - {str(e)}")
    
    logger.info(f"백업 정리 완료: {removed_count}개 파일 삭제됨")
    return removed_count