import time
import os
import pandas as pd
from django.core.management.base import BaseCommand
import requests as req
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from jobapp.models import JobPosting, JobDetail, JobSalary
from datetime import datetime
from django.conf import settings
import logging
import random
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import traceback

# 로깅 설정
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '잡코리아 채용 공고 크롤링'

    def add_arguments(self, parser):
        # 크롤링할 페이지 수를 인자로 받을 수 있게 추가, 총 86페이지
        parser.add_argument(
            '--pages',
            type=int,
            default=3,
            help='크롤링 할 페이지 수'
        )
        # 백업 파일 형식 선택 옵션 추가
        parser.add_argument(
            '--backup-format',
            type=str,
            default='csv',
            choices=['csv', 'excel', 'both'],
            help='백업 파일 형식: csv, excel, 또는 both'
        )

    # 재시도 기능을 하는 데코레이터 함수
    def retry_on_timeout(self, max_retries=3, retry_delay=5):
        def decorator(func):
            def wrapper(*args, **kwargs):
                retries = 0
                while retries < max_retries:
                    try:
                        return func(*args, **kwargs)
                    except (TimeoutException, WebDriverException) as e:
                        retries += 1
                        if retries >= max_retries:
                            self.stdout.write(self.style.ERROR(f"최대 재시도 횟수 초과: {str(e)}"))
                            raise
                        self.stdout.write(self.style.WARNING(f"연결 실패, {retry_delay}초 후 재시도 ({retries}/{max_retries}): {str(e)}"))
                        time.sleep(retry_delay)
            return wrapper
        return decorator

    def handle(self, *args, **kwargs):
        # 환경 변수에서 로그인 정보 가져오기 (보안 개선)
        username = os.environ.get('JOBKOREA_USERNAME', '')
        password = os.environ.get('JOBKOREA_PASSWORD', '')
        
        # 로그인 정보가 환경 변수에 없는 경우 처리
        if not username or not password:
            self.stdout.write(self.style.WARNING('환경 변수에 로그인 정보가 없습니다.'))
            username = getattr(settings, 'JOBKOREA_USERNAME', '')
            password = getattr(settings, 'JOBKOREA_PASSWORD', '')
            
            if not username or not password:
                self.stdout.write(self.style.ERROR('로그인 정보를 찾을 수 없습니다.'))
                return
        
        # 크롤링할 페이지 수
        pages_to_crawl = kwargs['pages']
        # 백업 파일 형식
        backup_format = kwargs['backup_format']
        
        # Selenium 설정
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # 헤드리스 모드 활성화 (백그라운드 실행)
        options.add_argument('--no-sandbox') # 샌드박스 모드 비활성화
        options.add_argument('--disable-dev-shm-usage') # 메모리 사용 최적화
        options.add_argument('--disable-gpu') # GPU 가속 비활성화 (헤드리스 모드에서 필요할 수 있음)
        options.add_argument('--window-size=1920,1080')
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
        options.add_argument('--dns-prefetch-disable')  # DNS 프리패치 비활성화
        options.page_load_strategy = 'eager'  # 페이지 로드 전략을 'eager'로 설정 (필수 리소스만 로드)
        
        # 새로운 타임아웃 설정 추가
        service = Service(ChromeDriverManager().install())
        driver = None
        
        try:
            # WebDriver 타임아웃 설정 추가
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(30)  # 페이지 로드 타임아웃 30초로 설정
            driver.set_script_timeout(30)     # 스크립트 실행 타임아웃 30초로 설정
            
            jobs_processed = 0
            job_data_list = []

            # 잡코리아 접속 및 로그인 (재시도 메커니즘 적용)
            self.stdout.write(self.style.SUCCESS('잡코리아 접속 시도 중...'))
            
            # 사이트 접속 (재시도 로직 적용)
            max_retries = 3
            retries = 0
            while retries < max_retries:
                try:
                    driver.get('https://www.jobkorea.co.kr/')
                    time.sleep(3)  # 로딩 시간 증가
                    break
                except (TimeoutException, WebDriverException) as e:
                    retries += 1
                    if retries >= max_retries:
                        raise Exception(f"사이트 접속 실패 (최대 재시도 횟수 초과): {str(e)}")
                    self.stdout.write(self.style.WARNING(f"사이트 접속 실패, 5초 후 재시도 ({retries}/{max_retries}): {str(e)}"))
                    time.sleep(5)

            # 로그인 시도
            self.stdout.write(self.style.SUCCESS('로그인 시도 중...'))
            try:
                # 로그인 버튼 찾기 - 선택자 업데이트
                try:
                    login_button = driver.find_element(By.CSS_SELECTOR, '.login > a')
                except NoSuchElementException:
                    # 대체 선택자 시도
                    login_button = driver.find_element(By.CSS_SELECTOR, '.btnLogin')
                login_button.click()
                time.sleep(2)
                
                driver.find_element(By.ID, 'M_ID').send_keys(username)
                driver.find_element(By.ID, 'M_PWD').send_keys(password)
                
                # 로그인 버튼 클릭 - 선택자 업데이트
                try:
                    driver.find_element(By.CSS_SELECTOR, '.login-button').click()
                except NoSuchElementException:
                    # 대체 선택자 시도
                    driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]').click()
                time.sleep(5)  # 로그인 후 대기 시간 증가
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'로그인 과정에서 오류 발생: {str(e)}'))
                raise

            # 채용정보로 이동 - 선택자 업데이트 및 다양한 접근 방식 시도
            try:
                # 첫 번째 방식 시도
                driver.find_element(By.CSS_SELECTOR, '#gnbGi > a > span > span').click()
            except NoSuchElementException:
                try:
                    # 두 번째 방식 시도
                    driver.find_element(By.XPATH, '//a[contains(text(), "채용정보")]').click()
                except NoSuchElementException:
                    # 직접 URL로 이동
                    driver.get('https://www.jobkorea.co.kr/recruit/joblist')
            time.sleep(5)  # 페이지 로딩 대기 시간 증가

            # 직무별 카테고리로 이동 - 다양한 접근 방식 시도
            try:
                # 첫 번째 방식 시도
                driver.find_element(By.CSS_SELECTOR, '#content > div.rcr_lnb > ul > li:nth-child(2) > a').click()
            except NoSuchElementException:
                try:
                    # 두 번째 방식 시도
                    driver.find_element(By.XPATH, '//a[contains(text(), "직무별")]').click()
                except NoSuchElementException:
                    # 직접 URL로 이동
                    driver.get('https://www.jobkorea.co.kr/recruit/joblist?menucode=duty')
            time.sleep(5)  # 페이지 로딩 대기 시간 증가

            # 관련 필터 선택 - 선택자 변경 가능성 고려 및 오류 처리 추가
            try:
                # IT/인터넷 카테고리 선택
                try:
                    # XPATH 방식으로 시도
                    driver.find_element(By.XPATH, '//*[@id="devSearchForm"]/div[2]/div/div[1]/dl[1]/dd[2]/div[2]/dl[1]/dd/div[1]/ul/li[6]/label/span/span').click()
                except NoSuchElementException:
                    # 텍스트 검색 방식으로 시도
                    it_elements = driver.find_elements(By.XPATH, '//span[contains(text(), "IT/인터넷")]')
                    if it_elements:
                        it_elements[0].click()
                    else:
                        raise Exception("IT/인터넷 카테고리를 찾을 수 없습니다.")
                time.sleep(3)
                
                # 개발자 선택 시도 - 여러 방식으로 시도
                dev_selectors = [
                    '#duty_step2_10031_ly > li:nth-child(1) > label > span > span',  # 백엔드 개발자
                    '#duty_step2_10031_ly > li:nth-child(2) > label > span > span',  # 빅데이터
                    '#duty_step2_10031_ly > li:nth-child(3) > label > span',         # 웹개발자
                    '#duty_step2_10031_ly > li:nth-child(14) > label > span > span'  # AI/ML엔지니어
                ]
                
                for selector in dev_selectors:
                    try:
                        driver.find_element(By.CSS_SELECTOR, selector).click()
                        time.sleep(1)  # 클릭 사이에 짧은 대기
                    except NoSuchElementException:
                        self.stdout.write(self.style.WARNING(f"선택자를 찾을 수 없습니다: {selector}"))
                        continue
                
                # 검색 버튼 클릭
                try:
                    driver.find_element(By.CSS_SELECTOR, '#dev-btn-search').click()
                except NoSuchElementException:
                    # 대체 검색 버튼 찾기
                    search_buttons = driver.find_elements(By.XPATH, '//button[contains(text(), "검색")]')
                    if search_buttons:
                        search_buttons[0].click()
                    else:
                        raise Exception("검색 버튼을 찾을 수 없습니다.")
                time.sleep(5)  # 검색 결과 로딩 대기 시간 증가
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'필터 선택 중 오류 발생: {str(e)}'))
                # 필터 선택 오류가 있더라도 계속 진행 (기본 목록 활용)
                pass

            # 페이지네이션 처리
            base_list_url = None  # 목록 페이지 기본 URL 저장 변수
            
            for page in range(1, pages_to_crawl + 1):
                self.stdout.write(self.style.SUCCESS(f'크롤링 {page}/{pages_to_crawl} 페이지'))
                time.sleep(3)  # 페이지 로딩 대기 시간 증가
                
                # 첫 페이지에서 기본 URL 저장 (이후 페이지 이동에 사용)
                if page == 1:
                    base_list_url = driver.current_url
                    self.stdout.write(self.style.SUCCESS(f'목록 페이지 URL: {base_list_url}'))
                
                if page > 1:
                    # 페이지네이션 요소 디버깅
                    self.stdout.write(self.style.SUCCESS(f'페이지 {page}로 이동 시도...'))
                    page_source = driver.page_source
                    pagination_soup = BeautifulSoup(page_source, 'html.parser')
                    
                    # 페이지네이션 요소 분석 (디버깅용)
                    pagination_elements = pagination_soup.select('.tplPagination a, .paging a, .pagination a, .page-item a, a.page-link')
                    self.stdout.write(self.style.SUCCESS(f'발견된 페이지네이션 요소: {len(pagination_elements)}개'))
                    for i, page_elem in enumerate(pagination_elements[:5]):  # 처음 5개만 출력
                        self.stdout.write(f"  페이지 요소 {i}: 텍스트={page_elem.get_text(strip=True)}, href={page_elem.get('href', '없음')}")
                    
                    # 페이지 이동 - 여러 방식 시도
                    page_moved = False
                    
                    # 방법 1: 저장된 기본 URL에 페이지 파라미터 추가
                    try:
                        if base_list_url:
                            # URL에 이미 페이지 파라미터가 있는지 확인
                            if '?' in base_list_url:
                                if 'page=' in base_list_url or 'Page=' in base_list_url or 'pg=' in base_list_url:
                                    # 기존 페이지 파라미터 교체
                                    import re
                                    page_url = re.sub(r'(page=|Page=|pg=)\d+', f'\\1{page}', base_list_url)
                                else:
                                    # 페이지 파라미터 추가
                                    page_url = f"{base_list_url}&page={page}"
                            else:
                                # 새 파라미터로 페이지 추가
                                page_url = f"{base_list_url}?page={page}"
                                
                            self.stdout.write(self.style.SUCCESS(f'URL로 이동 시도: {page_url}'))
                            driver.get(page_url)
                            time.sleep(5)  # 페이지 로딩 대기 시간 증가
                            page_moved = True
                    except Exception as e:
                        self.stdout.write(self.style.WARNING(f'URL 기반 페이지 이동 실패: {str(e)}'))
                    
                    # 방법 2: 다양한 CSS 선택자로 페이지 링크 클릭 시도
                    if not page_moved:
                        selectors = [
                            f'#anchorGICnt_{page}',                          # 기존 선택자 1
                            f'.tplPagination a[href*="Page={page}"]',        # 페이지 번호가 포함된 href
                            f'.paging a[href*="page={page}"]',               # 다른 형식의 페이지 파라미터
                            f'.pagination a[href*="page={page}"]',           # Bootstrap 스타일 페이지네이션
                            f'.page-item:nth-child({page + 1}) a',           # 페이지 항목 위치 기반 (0번이 첫 페이지일 경우)
                            f'a.page-link[href*="page={page}"]',             # 페이지 링크 클래스
                            f'a[data-page="{page}"]',                        # 데이터 속성 기반
                        ]
                        
                        for selector in selectors:
                            try:
                                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                                if elements:
                                    self.stdout.write(self.style.SUCCESS(f'선택자로 페이지 요소 발견: {selector}'))
                                    elements[0].click()
                                    time.sleep(5)  # 페이지 이동 후 대기 시간 증가
                                    page_moved = True
                                    break
                            except Exception as e:
                                self.stdout.write(self.style.WARNING(f'선택자 {selector} 클릭 실패: {str(e)}'))
                                continue
                    
                    # 방법 3: XPath로 페이지 번호 텍스트 포함 요소 클릭
                    if not page_moved:
                        try:
                            # 정확한 페이지 번호 일치 요소
                            page_number_xpath = f'//a[text()="{page}"]'
                            elements = driver.find_elements(By.XPATH, page_number_xpath)
                            
                            if not elements:
                                # 페이지 번호가 포함된 요소 (더 넓은 범위)
                                page_number_xpath = f'//a[contains(text(), "{page}")]'
                                elements = driver.find_elements(By.XPATH, page_number_xpath)
                            
                            if elements:
                                self.stdout.write(self.style.SUCCESS(f'XPath로 페이지 요소 발견: {page_number_xpath}'))
                                elements[0].click()
                                time.sleep(5)  # 페이지 이동 후 대기 시간 증가
                                page_moved = True
                        except Exception as e:
                            self.stdout.write(self.style.WARNING(f'XPath 페이지 이동 실패: {str(e)}'))
                    
                    # 모든 방법으로 이동 실패 시
                    if not page_moved:
                        self.stdout.write(self.style.ERROR(f'페이지 {page}로 이동할 수 없습니다. 크롤링을 종료합니다.'))
                        break

                # 페이지 소스 가져오기
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')

                # 채용 목록 파싱 - 다양한 CSS 선택자 시도
                job_items = soup.select('.list-default .list-post')
                
                # 기존 선택자로 찾지 못한 경우 대체 선택자 시도
                if not job_items:
                    self.stdout.write(self.style.WARNING('기존 선택자로 채용 공고를 찾을 수 없습니다. 대체 선택자 시도...'))
                    # 대체 선택자 목록
                    alternative_selectors = [
                        'div.recruit-info',           # 대체 선택자 1
                        'li.list-post',               # 대체 선택자 2
                        'div.job-list-item',          # 대체 선택자 3
                        'li.list-post-item',          # 대체 선택자 4 
                        'div.post',                   # 대체 선택자 5
                        'tr.devloopArea',             # 대체 선택자 6
                        '.jlist-box .list-default li',  # 대체 선택자 7
                        'div.recruit-info',           # 대체 선택자 8 
                        'div.list-post',              # 대체 선택자 9
                        '.devpager-wrap tbody tr',    # 대체 선택자 10
                        '.jlist-table-wrap tr',       # 대체 선택자 11
                    ]
                    
                    for selector in alternative_selectors:
                        job_items = soup.select(selector)
                        if job_items:
                            self.stdout.write(self.style.SUCCESS(f'대체 선택자 "{selector}"로 {len(job_items)}개의 채용 공고를 찾았습니다.'))
                            break
                
                # 여전히 채용 공고를 찾지 못한 경우
                if not job_items:
                    self.stdout.write(self.style.WARNING('채용 공고를 찾을 수 없습니다. 다음 페이지로 진행합니다.'))
                    continue
                
                # 디버깅을 위해 첫 번째 채용 공고 구조 출력
                if len(job_items) > 0:
                    first_job = job_items[0]
                    self.stdout.write(self.style.SUCCESS('채용 공고 HTML 구조 분석:'))
                    # 중요 자식 요소들 출력
                    for i, child in enumerate(first_job.find_all(['td', 'a', 'div', 'span'], recursive=False, limit=5)):
                        self.stdout.write(f"자식 요소 {i}: {child.name}, 클래스: {child.get('class', '없음')}")
                        # 각 요소의 자식들도 확인
                        for j, grandchild in enumerate(child.find_all(['a', 'div', 'span'], limit=3)):
                            self.stdout.write(f"  손자 요소 {j}: {grandchild.name}, 텍스트: {grandchild.get_text(strip=True)[:30]}...")

                for job_item in job_items:
                    try:
                        # tr.devloopArea 구조에 맞는 새로운 파싱 로직
                        if 'devloopArea' in job_item.get('class', []):
                            # 회사명 추출 - 사용자가 제공한 선택자 및 대체 선택자 시도
                            company_name = "알 수 없음"
                            
                            # 먼저 사용자가 제공한 정확한 선택자 시도 (페이지 전체에서)
                            company_selector = '#dev-gi-list > div > div.tplList.tplJobList > table > tbody > tr > td.tplCo > a'
                            company_elems = soup.select(company_selector)
                            
                            # 현재 처리 중인 행의 인덱스 찾기
                            current_index = -1
                            for i, item in enumerate(job_items):
                                if item == job_item:
                                    current_index = i
                                    break
                            
                            # 해당 인덱스에 맞는 회사명 요소 가져오기
                            if 0 <= current_index < len(company_elems):
                                company_name = company_elems[current_index].get_text(strip=True)
                            
                            # 위 방식으로 못 찾은 경우 대체 선택자 시도
                            if company_name == "알 수 없음":
                                # 대체 방법 1: td.tplCo 직접 찾기
                                company_td = job_item.find('td', class_='tplCo')
                                if company_td and company_td.find('a'):
                                    company_name = company_td.find('a').get_text(strip=True)
                                
                                # 대체 방법 2: 일반적인 회사명 패턴 찾기
                                if company_name == "알 수 없음":
                                    company_td = job_item.find('td', class_=lambda c: c and ('cell_first' in c or 'company' in c or 'corp' in c or 'tplCo' in c))
                                    if company_td:
                                        company_a = company_td.find('a')
                                        if company_a and company_a.get_text(strip=True):
                                            company_name = company_a.get_text(strip=True)
                                        elif company_td.get_text(strip=True):
                                            company_name = company_td.get_text(strip=True)
                            
                            # 채용 타이틀 (일반적으로 두 번째 td에 있는 a 태그)
                            title_td = job_item.find('td', class_='tplTit')
                            if not title_td:
                                title_td = job_item.find_all('td')[1] if len(job_item.find_all('td')) > 1 else None
                            
                            title_a = None
                            if title_td:
                                title_a = title_td.find('a')
                            
                            job_title = "제목 없음"
                            if title_a and title_a.get_text(strip=True):
                                job_title = title_a.get_text(strip=True)
                            elif title_td and title_td.get_text(strip=True):
                                job_title = title_td.get_text(strip=True)
                            
                            # 상세 페이지 URL
                            detail_url = None
                            if title_a and 'href' in title_a.attrs:
                                href = title_a['href']
                                # 상대 URL을 절대 URL로 변환
                                if href.startswith('/'):
                                    detail_url = 'https://www.jobkorea.co.kr' + href
                                else:
                                    detail_url = href
                            
                            # 기본 정보 추출 (경력, 학력, 위치, 고용형태)
                            # 일반적으로 3번째, 4번째 TD에 위치
                            all_tds = job_item.find_all('td')
                            
                            experience = None
                            education = None
                            location = None
                            employment_type = None
                            
                            # 경력 정보
                            experience_td = job_item.find('td', class_='tplEdu')
                            if not experience_td and len(all_tds) > 2:
                                experience_td = all_tds[2]
                            if experience_td and experience_td.get_text(strip=True):
                                experience = experience_td.get_text(strip=True)
                            
                            # 학력 정보
                            education_td = job_item.find('td', class_='tplGrd')
                            if not education_td and len(all_tds) > 3:
                                education_td = all_tds[3]
                            if education_td and education_td.get_text(strip=True):
                                education = education_td.get_text(strip=True)
                            
                            # 지역 정보
                            location_td = job_item.find('td', class_='tplLoc')
                            if not location_td and len(all_tds) > 4:
                                location_td = all_tds[4]
                            if location_td and location_td.get_text(strip=True):
                                location = location_td.get_text(strip=True)
                            
                            # 고용형태
                            employment_td = job_item.find('td', class_='tplWor')
                            if not employment_td and len(all_tds) > 5:
                                employment_td = all_tds[5]
                            if employment_td and employment_td.get_text(strip=True):
                                employment_type = employment_td.get_text(strip=True)
                            
                        else:
                            # 기존 파싱 방식도 시도
                            try:
                                # 기존 방식 시도
                                company_elem = job_item.select_one('.post-list-corp a')
                                company_name = company_elem.text.strip() if company_elem else "알 수 없음"
                                
                                job_title_elem = job_item.select_one('.post-list-info .title')
                                job_title = job_title_elem.text.strip() if job_title_elem else "제목 없음"
                                
                                detail_url_elem = job_item.select_one('.post-list-info .title a')
                                detail_url = 'https://www.jobkorea.co.kr' + detail_url_elem['href'] if detail_url_elem and 'href' in detail_url_elem.attrs else None
                                
                                info_elem = job_item.select_one('.post-list-info .option')
                                
                                experience = None
                                education = None
                                location = None
                                employment_type = None
                                
                                if info_elem:
                                    info_spans = info_elem.select('span')
                                    if len(info_spans) >= 1:
                                        experience = info_spans[0].text.strip()
                                    if len(info_spans) >= 2:
                                        education = info_spans[1].text.strip()
                                    if len(info_spans) >= 3:
                                        location = info_spans[2].text.strip()
                                    if len(info_spans) >= 4:
                                        employment_type = info_spans[3].text.strip()
                            except Exception as e:
                                self.stdout.write(self.style.WARNING(f"기존 파싱 방식 실패: {str(e)}"))
                        
                        # 필수 데이터 검증
                        if not company_name or not job_title or not detail_url:
                            self.stdout.write(self.style.WARNING(f"필수 데이터 누락: 회사={company_name}, 제목={job_title}, URL={detail_url}"))
                            continue
                        
                        # 상세 페이지로 이동
                        try:
                            # 상세 페이지 로드 (재시도 로직 적용)
                            retries_detail = 0
                            max_retries_detail = 3
                            while retries_detail < max_retries_detail:
                                try:
                                    driver.get(detail_url)
                                    time.sleep(random.uniform(1.5, 3))  # 무작위 대기 시간 (서버 부하 방지)
                                    break
                                except (TimeoutException, WebDriverException) as e:
                                    retries_detail += 1
                                    if retries_detail >= max_retries_detail:
                                        raise Exception(f"상세 페이지 접속 실패 (최대 재시도 횟수 초과): {str(e)}")
                                    self.stdout.write(self.style.WARNING(f"상세 페이지 접속 실패, 5초 후 재시도 ({retries_detail}/{max_retries_detail}): {str(e)}"))
                                    time.sleep(5)
                                    
                            detail_source = driver.page_source
                            detail_soup = BeautifulSoup(detail_source, 'html.parser')
                        except Exception as e:
                            self.stdout.write(self.style.WARNING(f"상세 페이지 로드 중 오류: {str(e)}, 건너뜁니다"))
                            continue
                        
                        # 상세 정보를 담을 딕셔너리
                        job_details = {
                            'description': None,
                            'job_field': None,
                            'recruitment_count': None,
                            'experience': experience,
                            'education': education,
                            'location': location,
                            'employment_type': employment_type,
                            'skills': None,
                        }
                        
                        # 상세 설명 추출 시도 - 여러 선택자 시도
                        description_selectors = [
                            '.jv_cont .jv_summary .cont',
                            '.jobs-statement',
                            '.jobpost-user-content',
                            '.view-section-wrap > .view-section:nth-child(1) .view-paragraph',
                            '.job-description',
                            '.tbRow.clear > .tbCol:nth-child(1) > .col > dd',
                        ]
                        
                        for selector in description_selectors:
                            description_elem = detail_soup.select_one(selector)
                            if description_elem:
                                job_details['description'] = description_elem.get_text(strip=True)
                                break
                        
                        # 정보를 담고 있는 dl/dt/dd 구조를 찾아서 파싱
                        all_dt_elements = detail_soup.select('dt, th')
                        for dt in all_dt_elements:
                            dt_text = dt.get_text(strip=True)
                            
                            # 다음 dd 또는 td 요소 찾기
                            next_value = dt.find_next('dd') or dt.find_next('td')
                            if not next_value:
                                continue
                                
                            value_text = next_value.get_text(strip=True)
                            
                            # 각 정보 카테고리별 파싱
                            if '경력' in dt_text or '경험' in dt_text:
                                if not job_details['experience']:  # 리스트 페이지에서 가져오지 못한 경우
                                    job_details['experience'] = value_text
                            
                            elif '학력' in dt_text or '학위' in dt_text:
                                if not job_details['education']:
                                    job_details['education'] = value_text
                            
                            elif '지역' in dt_text or '위치' in dt_text or '근무지' in dt_text:
                                if not job_details['location']:
                                    job_details['location'] = value_text
                            
                            elif '고용형태' in dt_text or '계약' in dt_text or '근무형태' in dt_text:
                                if not job_details['employment_type']:
                                    job_details['employment_type'] = value_text
                            
                            elif '직무' in dt_text or '모집분야' in dt_text or '직종' in dt_text:
                                job_details['job_field'] = value_text
                            
                            elif '인원' in dt_text or '모집인원' in dt_text:
                                try:
                                    # 숫자만 추출
                                    import re
                                    numbers = re.findall(r'\d+', value_text)
                                    if numbers:
                                        job_details['recruitment_count'] = int(numbers[0])
                                except ValueError:
                                    pass
                            
                            elif '스킬' in dt_text or '기술' in dt_text or '우대사항' in dt_text:
                                job_details['skills'] = value_text  # required_skills -> skills로 변경
                        
                        # 스킬 요구사항 특별 처리 (일반적으로 별도 섹션에 있음)
                        if not job_details['skills']:
                            skill_selectors = [
                                '.jv_benefit',
                                '.jobpost-requirements',
                                '.preferred-skills',
                                '.view-section:contains("우대사항")',
                                '.view-section:contains("자격요건")',
                                '.view-section:contains("기술스택")'
                            ]
                            
                            for selector in skill_selectors:
                                try:
                                    # CSS 선택자가 :contains를 포함하는 경우 직접 처리
                                    if ':contains' in selector:
                                        base_selector, contains_text = selector.split(':contains')
                                        contains_text = contains_text.strip('(")')
                                        
                                        # 모든 관련 섹션 찾기
                                        for section in detail_soup.select(base_selector):
                                            if contains_text in section.get_text():
                                                job_details['skills'] = section.get_text(strip=True)  # required_skills -> skills로 변경
                                                break
                                    else:
                                        # 일반 선택자
                                        skill_elem = detail_soup.select_one(selector)
                                        if skill_elem:
                                            job_details['skills'] = skill_elem.get_text(strip=True)  # required_skills -> skills로 변경
                                            break
                                except Exception as e:
                                    self.stdout.write(self.style.WARNING(f"스킬 정보 파싱 중 오류: {str(e)}"))
                        
                        # 책임업무(responsibilities) 추출 시도
                        responsibility_selectors = [
                            '.view-section:contains("주요업무")',
                            '.view-section:contains("담당업무")',
                            '.jobpost-responsibilities',
                            '.job-responsibilities'
                        ]
                        
                        # 책임업무 정보 추출
                        for selector in responsibility_selectors:
                            try:
                                if ':contains' in selector:
                                    base_selector, contains_text = selector.split(':contains')
                                    contains_text = contains_text.strip('(")')
                                    
                                    # 모든 관련 섹션 찾기
                                    for section in detail_soup.select(base_selector):
                                        if contains_text in section.get_text():
                                            job_details['responsibilities'] = section.get_text(strip=True)
                                            break
                                else:
                                    # 일반 선택자
                                    resp_elem = detail_soup.select_one(selector)
                                    if resp_elem:
                                        job_details['responsibilities'] = resp_elem.get_text(strip=True)
                                        break
                            except Exception as e:
                                self.stdout.write(self.style.WARNING(f"책임업무 정보 파싱 중 오류: {str(e)}"))
                        
                        # DB에 저장
                        job_posting, created = JobPosting.objects.update_or_create(
                            company_name=company_name,
                            job_title=job_title,
                            defaults={
                                'employment_type': job_details['employment_type'],
                                'recruitment_count': job_details['recruitment_count'],
                                'location': job_details['location'],
                                'detail_url': detail_url,
                            }
                        )
                        
                        # 상세 정보 저장 - 필드명 매핑 수정
                        JobDetail.objects.update_or_create(
                            job_posting=job_posting,
                            defaults={
                                'job_field': job_details['job_field'],
                                'description': job_details['description'],
                                'career_level': job_details['experience'],
                                'education_level': job_details['education'],
                                'skills': job_details.get('skills'),  # required_skills -> skills로 변경
                                'responsibilities': job_details.get('responsibilities')  # 책임업무 추가
                            }
                        )
                        
                        # 백업을 위한 데이터 저장 - 필드명 매핑 수정
                        job_data = {
                            'company_name': company_name,
                            'job_title': job_title,
                            'detail_url': detail_url,
                            'employment_type': job_details['employment_type'],
                            'recruitment_count': job_details['recruitment_count'],
                            'location': job_details['location'],
                            'job_field': job_details['job_field'],
                            'description': job_details['description'],
                            'career_level': job_details['experience'],
                            'education_level': job_details['education'],
                            'skills': job_details.get('skills'),  # required_skills -> skills로 변경
                            'responsibilities': job_details.get('responsibilities'),  # 책임업무 추가
                            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        job_data_list.append(job_data)
                        
                        status = '새로 추가됨' if created else '업데이트됨'
                        self.stdout.write(self.style.SUCCESS(f'[{status}] {company_name} - {job_title}'))
                        jobs_processed += 1
                        
                        # 상세 페이지 처리 후 목록 페이지로 돌아가기
                        try:
                            if page == 1:
                                # 첫 페이지의 경우 기본 저장된 URL로 돌아감
                                driver.get(base_list_url)
                            else:
                                # 현재 처리 중인 페이지 URL로 돌아감 (페이지 파라미터 포함)
                                current_page_url = driver.current_url if not base_list_url else base_list_url
                                if '?' in current_page_url:
                                    if 'page=' in current_page_url or 'Page=' in current_page_url or 'pg=' in current_page_url:
                                        # 기존 페이지 파라미터 교체
                                        import re
                                        current_page_url = re.sub(r'(page=|Page=|pg=)\d+', f'\\1{page}', current_page_url)
                                    else:
                                        # 페이지 파라미터 추가
                                        current_page_url = f"{current_page_url}&page={page}"
                                else:
                                    # 새 파라미터로 페이지 추가
                                    current_page_url = f"{current_page_url}?page={page}"
                                
                                driver.get(current_page_url)
                            time.sleep(3)  # 페이지 로딩 대기 시간
                            
                            # 목록 페이지로 돌아온 후 현재 상태 확인 (디버깅)
                            self.stdout.write(self.style.SUCCESS(f'목록 페이지로 돌아옴: {driver.current_url}'))
                        except Exception as e:
                            self.stdout.write(self.style.WARNING(f'목록 페이지로 돌아가는 중 오류: {str(e)}'))
                            # 오류 발생 시 기본 URL로 복귀 시도
                            try:
                                if base_list_url:
                                    driver.get(base_list_url)
                                    time.sleep(3)
                            except Exception as e2:
                                self.stdout.write(self.style.ERROR(f'복구 시도 중 오류: {str(e2)}'))
                        
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f'채용 정보 파싱 중 오류: {str(e)}'))
                        # 디버깅을 위한 예외 상세 정보 출력
                        self.stdout.write(self.style.ERROR(f'예외 상세: {traceback.format_exc()}'))
                        continue
                
                # 페이지 간 대기 시간 (서버 부하 방지)
                time.sleep(random.uniform(2, 4))

            # 크롤링 결과를 파일로 저장
            if job_data_list:
                self._save_to_files(job_data_list, backup_format)
                
            self.stdout.write(self.style.SUCCESS(f'크롤링 완료: {jobs_processed}개 채용공고 처리됨'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'크롤링 중 오류 발생: {str(e)}'))
            # 더 자세한 에러 메시지 및 스택 트레이스 로깅
            logger.error(f"크롤링 오류 상세: {traceback.format_exc()}")

        finally:
            # 항상 드라이버 종료 보장
            if driver:
                try:
                    self.stdout.write(self.style.SUCCESS('브라우저 세션 종료 중...'))
                    driver.quit()
                except Exception as close_error:
                    self.stdout.write(self.style.ERROR(f'브라우저 종료 중 오류: {close_error}'))
    
    def _save_to_files(self, job_data_list, backup_format):
        """크롤링한 데이터를 파일로 저장"""
        # 백업 디렉토리 생성
        backup_dir = os.path.join(settings.BASE_DIR, 'backup_data')
        os.makedirs(backup_dir, exist_ok=True)
        
        # 현재 날짜와 시간으로 파일명 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # DataFrame 생성
        df = pd.DataFrame(job_data_list)
        
        # CSV 파일로 저장
        if backup_format in ['csv', 'both']:
            csv_path = os.path.join(backup_dir, f'job_data_{timestamp}.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig는 Excel에서 한글 깨짐 방지
            self.stdout.write(self.style.SUCCESS(f'CSV 파일로 저장됨: {csv_path}'))
        
        # Excel 파일로 저장
        if backup_format in ['excel', 'both']:
            excel_path = os.path.join(backup_dir, f'job_data_{timestamp}.xlsx')
            df.to_excel(excel_path, index=False, engine='openpyxl')
            self.stdout.write(self.style.SUCCESS(f'Excel 파일로 저장됨: {excel_path}'))