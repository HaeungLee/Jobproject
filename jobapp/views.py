from django.shortcuts import render, redirect
from django.db.models import Count, Avg, Q, Max, Min
from django.http import JsonResponse
from .models import JobPosting, JobDetail, CompanyInfo, JobSalary
import json
from collections import Counter
import re
import os
import glob
import pandas as pd
from django.conf import settings
from datetime import datetime
import numpy as np
from .ml_models import JobClustering, SalaryPredictor, TrendPredictor, JobFieldClassifier, train_all_models, GemmaModel
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import joblib
from django.contrib import messages
import requests
import koreanize_matplotlib

def home(request):
    """메인 페이지 뷰"""
    total_jobs = JobPosting.objects.count()
    recent_jobs = JobPosting.objects.order_by('-created_at')[:5]
    
    # 백업 파일 수
    backup_dir = os.path.join(settings.BASE_DIR, 'backup_data')
    csv_count = len(glob.glob(os.path.join(backup_dir, '*.csv')))
    excel_count = len(glob.glob(os.path.join(backup_dir, '*.xlsx')))
    backup_count = csv_count + excel_count
    
    context = {
        'total_jobs': total_jobs,
        'recent_jobs': recent_jobs,
        'backup_count': backup_count
    }
    return render(request, 'jobapp/home.html', context)

def job_list(request):
    """채용 공고 목록 페이지"""
    jobs = JobPosting.objects.all().order_by('-created_at')
    
    # 필터링 옵션
    location_filter = request.GET.get('location', '')
    if location_filter:
        jobs = jobs.filter(location__contains=location_filter)
        
    employment_type_filter = request.GET.get('employment_type', '')
    if employment_type_filter:
        jobs = jobs.filter(employment_type__contains=employment_type_filter)
    
    context = {
        'jobs': jobs,
        'locations': JobPosting.objects.values_list('location', flat=True).distinct(),
        'employment_types': JobPosting.objects.values_list('employment_type', flat=True).distinct()
    }
    return render(request, 'jobapp/job_list.html', context)

def job_detail(request, job_id):
    """채용 공고 상세 페이지"""
    job = JobPosting.objects.get(id=job_id)
    
    try:
        job_detail = job.details
    except JobDetail.DoesNotExist:
        job_detail = None
        
    try:
        salary_info = job.salary
    except JobSalary.DoesNotExist:
        salary_info = None
    
    context = {
        'job': job,
        'job_detail': job_detail,
        'salary_info': salary_info
    }
    return render(request, 'jobapp/job_detail.html', context)

def generate_insights_from_db():
    """DB 데이터에서 동적 인사이트 생성"""
    # 불필요한 키워드 및 필터링할 항목들
    irrelevant_keywords = ['홈페이지 지원', '스크랩', '인쇄', '공유', '신고', '미리보기', '지원하기', '마감', '접수양식', '접수방법']
    
    # 트렌드 인사이트 생성
    job_fields_query = JobDetail.objects.values('job_field').annotate(count=Count('id')).order_by('-count')
    
    # 유효한 직무 필드만 필터링
    valid_job_fields = []
    for field in job_fields_query:
        if field['job_field'] and not any(keyword in field['job_field'] for keyword in irrelevant_keywords):
            # 단어 길이가 너무 긴 경우도 필터링 (비정상적인 데이터일 가능성)
            if len(field['job_field']) < 30:
                valid_job_fields.append(field)
    
    top_fields = [field['job_field'] for field in valid_job_fields[:3] if field['job_field']]
    trend_insight = f"현재 개발자 시장에서 가장 수요가 높은 분야는 {', '.join(top_fields)} 직무입니다."
    
    # 지역별 인사이트 생성 - 유효한 지역만 필터링
    locations_query = JobPosting.objects.values('location').annotate(count=Count('id')).order_by('-count')
    valid_locations = []
    
    for loc in locations_query:
        # None, 빈 문자열, '미지정' 제외하고 유효한 지역만 포함
        if loc['location'] and loc['location'].strip() and loc['location'] != '미지정':
            # 단어 길이가 비정상적으로 긴 경우 제외
            if len(loc['location']) < 20:
                valid_locations.append(loc)
    
    top_locations = [loc['location'] for loc in valid_locations[:3] if loc['location']]
    location_insight = f"대부분의 개발자 채용은 {', '.join(top_locations)} 지역에 집중되어 있습니다."
    
    # 경력 요구사항 인사이트 생성 - 유효한 경력 데이터만 필터링
    irrelevant_career_keywords = ['스크랩', '인쇄', '미리보기', '신고', '지원']
    
    career_levels_query = JobDetail.objects.values('career_level').annotate(count=Count('id')).order_by('-count')
    valid_career_levels = []
    
    for level in career_levels_query:
        if level['career_level'] and not any(keyword in level['career_level'] for keyword in irrelevant_career_keywords):
            # 단어 길이가 비정상적으로 긴 경우 제외
            if len(level['career_level']) < 20:
                valid_career_levels.append(level)
    
    top_career = next((level['career_level'] for level in valid_career_levels if level['career_level']), "경력무관")
    new_grad_count = JobDetail.objects.filter(career_level__icontains='신입').count()
    exp_count = JobDetail.objects.filter(career_level__icontains='경력').count()
    
    if exp_count > new_grad_count:
        #career_insight = f"경력직에 대한 수요가 신입보다 {round(exp_count/max(new_grad_count, 1), 1)}배 높으며, 특히 '{top_career}' 경력을 가진 개발자의 수요가 가장 많습니다."
        career_insight = f"'{top_career}' 경력을 가진 개발자의 수요가 가장 많습니다."
    else:
        career_insight = f"신입 개발자에 대한 수요가 높으며, 특히 '{top_career}' 포지션이 많습니다."
    
    # 학력 요구사항 인사이트 생성 - 유효한 학력 데이터만 필터링
    education_levels_query = JobDetail.objects.values('education_level').annotate(count=Count('id')).order_by('-count')
    valid_education_levels = []
    
    for level in education_levels_query:
        if level['education_level'] and not any(keyword in level['education_level'] for keyword in irrelevant_keywords):
            # 단어 길이가 비정상적으로 긴 경우 제외
            if len(level['education_level']) < 20:
                valid_education_levels.append(level)
    
    top_education = next((level['education_level'] for level in valid_education_levels if level['education_level']), "학력무관")
    
    if "무관" in top_education:
        education_insight = "대부분의 개발 직군은 학력보다 기술과 경험을 중요시하는 경향이 있습니다."
    else:
        education_insight = f"개발 직군에서 가장 많이 요구하는 학력은 '{top_education}'입니다."
    
    # 기술 스택 인사이트 생성 - 실제 기술 스택만 포함하도록 필터링
    skill_counter = Counter()
    common_tech_keywords = ['python', 'java', 'javascript', 'react', 'node', 'django', 'spring', 'mysql', 
                            'postgresql', 'mongodb', 'aws', 'docker', 'kubernetes', 'git', 'linux', 'html', 
                            'css', 'vue', 'angular', 'typescript', 'php', 'c#', 'c++', 'swift', 'kotlin', 
                            'go', 'rust', 'ruby', 'flask', 'laravel', 'tensorflow', 'pytorch', 'pandas', 
                            'numpy', 'scikit-learn', 'hadoop', 'spark', 'jenkins', 'ci/cd', 'graphql', 'rest']
    
    # 스킬 필터링 함수
    def is_valid_skill(skill):
        skill = skill.lower()
        # 너무 짧은 단어는 무시
        if len(skill) <= 1:
            return False
        # 흔한 무의미한 단어 필터링
        if skill in ['및', '등', '을', '를', '이', '가', '은', '는', '수', '하다', '있다', '없다', '함', '또한', '모두']:
            return False
        # 숫자만 있는 경우 무시
        if re.match(r'^\d+$', skill):
            return False
        # 일반적인 기술 키워드 포함 여부
        if any(tech in skill for tech in common_tech_keywords):
            return True
        # 길이가 긴 경우는 유효하지 않은 스킬일 가능성이 높음
        if len(skill) > 15:
            return False
        return True
    
    for detail in JobDetail.objects.all():
        if detail.skills:
            # 스킬 분리 (쉼표, 공백 등으로 구분)
            skills = re.split(r'[,\s]', detail.skills)
            for skill in skills:
                skill = skill.strip().lower()
                if is_valid_skill(skill):
                    skill_counter[skill] += 1
    
    # 상위 4개 스킬
    top_skills = [skill for skill, _ in skill_counter.most_common(4) if skill not in irrelevant_keywords]
    skill_insight = f"{', '.join(top_skills)}가 가장 많이 요구되는 기술 스택입니다."
    
    return {
        'trend_insight': trend_insight,
        'location_insight': location_insight,
        'career_insight': career_insight,
        'education_insight': education_insight,
        'skill_insight': skill_insight
    }

def job_insights(request):
    """채용 인사이트 페이지"""
    # 데이터 소스 선택 (DB 또는 백업 파일)
    data_source = request.GET.get('data_source', 'db')
    file_path = request.GET.get('file_path', '')
    
    # 가능한 백업 파일 목록 가져오기
    backup_files = []
    backup_dir = os.path.join(settings.BASE_DIR, 'backup_data')
    if os.path.exists(backup_dir):
        csv_files = glob.glob(os.path.join(backup_dir, '*.csv'))
        excel_files = glob.glob(os.path.join(backup_dir, '*.xlsx'))
        
        for f in sorted(csv_files + excel_files, reverse=True):  # 최신 파일이 먼저 오도록 정렬
            file_name = os.path.basename(f)
            file_size = os.path.getsize(f) / 1024  # KB 단위
            backup_files.append({
                'path': f,
                'name': file_name,
                'size': f"{file_size:.1f} KB",
                'date': datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')
            })
    
    if data_source == 'file' and file_path:
        # 백업 파일에서 데이터 로드
        job_data = load_data_from_file(file_path)
        
        # 차트 데이터 생성 및 컨텍스트에 추가
        chart_data = generate_chart_data_from_dataframe(job_data)
        context = chart_data  # 차트 JSON 데이터 설정
        context['data_source'] = '백업 파일'
        context['file_name'] = os.path.basename(file_path)
        
        # Matplotlib/Seaborn 그래프 이미지 생성 및 추가
        context.update(generate_matplotlib_graphs_from_dataframe(job_data))
        
        # 총 채용 공고 수 추가
        context['total_jobs'] = len(job_data)
    else:
        # DB에서 데이터 로드 (기존 코드 사용)
        # 전체 채용 수
        total_jobs = JobPosting.objects.count()
        
        # 불필요한 키워드 및 필터링할 항목들
        irrelevant_keywords = ['홈페이지 지원', '스크랩', '인쇄', '공유', '신고', '미리보기', '지원하기', '마감', '접수양식', '접수방법']
        
        # 직무 분야별 채용 분포
        job_field_data = {}
        job_details = JobDetail.objects.all()
        
        # 직무 분야 통계 (직무 분야가 여러 개 포함된 경우 각각 카운트)
        field_counter = Counter()
        for detail in job_details:
            if detail.job_field:
                # 쉼표나 /로 구분된 경우 분리
                fields = re.split(r'[,/]', detail.job_field)
                for field in fields:
                    field = field.strip()
                    # 유효한 필드만 처리 (너무 길거나 불필요한 키워드 포함된 경우 제외)
                    if field and len(field) < 30 and not any(keyword in field for keyword in irrelevant_keywords):
                        field_counter[field] += 1
        
        # 상위 10개 직무 분야만 사용
        top_fields = field_counter.most_common(10)
        job_field_data = {
            'labels': [field[0] for field in top_fields],
            'data': [field[1] for field in top_fields]
        }
        
        # 지역별 채용 현황 - 유효한 지역만 필터링
        location_stats = JobPosting.objects.values('location').annotate(count=Count('id')).order_by('-count')
        filtered_locations = []
        
        for loc in location_stats:
            # None, 빈 문자열 또는 '미지정' 필터링
            loc_value = loc['location'] if loc['location'] else '미지정'
            
            # 지역명이 너무 길거나 불필요한 키워드가 포함된 경우 제외
            if loc_value != '미지정' and len(loc_value) < 20 and not any(keyword in loc_value for keyword in irrelevant_keywords):
                filtered_locations.append({
                    'location': loc_value,
                    'count': loc['count']
                })
        
        # 상위 10개 지역만 사용
        top_locations = filtered_locations[:10]
        # 데이터가 너무 적으면 '기타 지역'으로 묶음
        if len(top_locations) < 3:
            top_locations = list(location_stats[:10])
        
        location_data = {
            'labels': [item['location'] or '미지정' for item in top_locations],
            'data': [item['count'] for item in top_locations]
        }
        
        # 경력 수준별 분포 - 유효한 경력 레벨만 필터링
        career_levels_query = JobDetail.objects.values('career_level').annotate(count=Count('id')).order_by('-count')
        filtered_career_levels = []
        
        for level in career_levels_query:
            career_value = level['career_level'] if level['career_level'] else '미지정'
            
            # 경력 설명이 너무 길거나 불필요한 키워드가 포함된 경우 제외
            if len(career_value) < 20 and not any(keyword in career_value for keyword in irrelevant_keywords):
                filtered_career_levels.append({
                    'career_level': career_value,
                    'count': level['count']
                })
        
        career_data = {
            'labels': [item['career_level'] for item in filtered_career_levels],
            'data': [item['count'] for item in filtered_career_levels]
        }
        
        # 학력 요구사항 분포 - 유효한 학력 데이터만 필터링
        education_levels_query = JobDetail.objects.values('education_level').annotate(count=Count('id')).order_by('-count')
        filtered_education_levels = []
        
        for level in education_levels_query:
            edu_value = level['education_level'] if level['education_level'] else '미지정'
            
            # 학력 설명이 너무 길거나 불필요한 키워드가 포함된 경우 제외
            if len(edu_value) < 20 and not any(keyword in edu_value for keyword in irrelevant_keywords):
                filtered_education_levels.append({
                    'education_level': edu_value,
                    'count': level['count']
                })
        
        education_data = {
            'labels': [item['education_level'] for item in filtered_education_levels],
            'data': [item['count'] for item in filtered_education_levels]
        }
        
        # AI/빅데이터 관련 채용 추세
        ai_jobs = JobDetail.objects.filter(
            Q(job_field__icontains='AI') | 
            Q(job_field__icontains='인공지능') |
            Q(job_field__icontains='머신러닝') |
            Q(job_field__icontains='딥러닝') |
            Q(job_field__icontains='빅데이터')
        ).count()
        
        # 웹/앱 개발 관련 채용 추세
        web_jobs = JobDetail.objects.filter(
            Q(job_field__icontains='웹') | 
            Q(job_field__icontains='프론트엔드') |
            Q(job_field__icontains='백엔드') |
            Q(job_field__icontains='풀스택')
        ).count()
        
        # 모바일 앱 개발 관련 채용
        app_jobs = JobDetail.objects.filter(
            Q(job_field__icontains='모바일') | 
            Q(job_field__icontains='앱') |
            Q(job_field__icontains='안드로이드') |
            Q(job_field__icontains='iOS')
        ).count()
        
        # 개발 분야별 채용 비교 데이터
        dev_field_comparison = {
            'labels': ['AI/빅데이터', '웹 개발', '앱 개발'],
            'data': [ai_jobs, web_jobs, app_jobs]
        }
        
        # 필요 기술 분석 - 유효한 기술 스택만 필터링
        skill_counter = Counter()
        
        # 자주 사용되는 개발 기술 키워드
        common_tech_keywords = ['python', 'java', 'javascript', 'react', 'node', 'django', 'spring', 'mysql', 
                               'postgresql', 'mongodb', 'aws', 'docker', 'kubernetes', 'git', 'linux', 'html', 
                               'css', 'vue', 'angular', 'typescript', 'php', 'c#', 'c++', 'swift', 'kotlin', 
                               'go', 'rust', 'ruby', 'flask', 'laravel', 'tensorflow', 'pytorch', 'pandas', 
                               'numpy', 'scikit-learn', 'hadoop', 'spark', 'jenkins', 'ci/cd', 'graphql', 'rest']
        
        # 스킬 필터링 함수
        def is_valid_skill(skill):
            skill = skill.lower()
            # 너무 짧은 단어는 무시
            if len(skill) <= 1:
                return False
            # 흔한 무의미한 단어 필터링
            if skill in ['및', '등', '을', '를', '이', '가', '은', '는', '수', '하다', '있다', '없다', '함', '또한', '모두']:
                return False
            # 숫자만 있는 경우 무시
            if re.match(r'^\d+$', skill):
                return False
            # 길이가 너무 긴 경우는 유효하지 않은 스킬일 가능성이 높음
            if len(skill) > 15:
                return False
            # 불필요한 키워드 포함 여부
            if any(keyword in skill for keyword in irrelevant_keywords):
                return False
            return True
        
        for detail in job_details:
            if detail.skills:
                # 스킬 분리 (쉼표, 공백 등으로 구분)
                skills = re.split(r'[,\s]', detail.skills)
                for skill in skills:
                    skill = skill.strip().lower()
                    if is_valid_skill(skill):
                        # 유효한 기술 스택인 경우만 카운트
                        if skill in common_tech_keywords or any(tech in skill for tech in common_tech_keywords):
                            skill_counter[skill] += 1
        
        # 상위 15개 스킬만 사용
        top_skills = skill_counter.most_common(15)
        skill_data = {
            'labels': [skill[0] for skill in top_skills],
            'data': [skill[1] for skill in top_skills]
        }
        
        context = {
            'total_jobs': total_jobs,
            'job_field_data_json': json.dumps(job_field_data),
            'location_data_json': json.dumps(location_data),
            'career_data_json': json.dumps(career_data),
            'education_data_json': json.dumps(education_data),
            'dev_field_comparison_json': json.dumps(dev_field_comparison),
            'skill_data_json': json.dumps(skill_data),
            'data_source': 'DB'
        }
        
        # 데이터 기반 인사이트 요약 생성
        context.update(generate_insights_from_db())
    
    # 백업 파일 목록 추가
    context['backup_files'] = backup_files
    
    # Matplotlib/Seaborn 그래프 이미지 생성 및 추가
    if data_source == 'file' and file_path:
        context.update(generate_matplotlib_graphs_from_dataframe(job_data))
    else:
        context.update(generate_matplotlib_graphs_from_db())
    
    return render(request, 'jobapp/insights.html', context)

def load_data_from_file(file_path):
    """백업 파일에서 데이터 로드"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    return pd.DataFrame()  # 빈 DataFrame 반환

def generate_matplotlib_graphs_from_db():
    """DB 데이터로부터 Matplotlib 그래프 생성 - 데이터 전처리 추가"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
    from matplotlib.ticker import MaxNLocator
    
    plt.switch_backend('Agg')  # 비대화형 백엔드 사용
    
    graph_data = {}
    
    # 1. 시간별 채용 공고 추이 (최근 2주)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 최근 2주간 날짜별 채용 공고 수 집계
    date_counts = JobPosting.objects.values('created_at__date').annotate(count=Count('id')).order_by('created_at__date')
    dates = [item['created_at__date'] for item in date_counts]
    counts = [item['count'] for item in date_counts]
    
    if dates:
        ax.plot(dates, counts, marker='o', linestyle='-', color='#4e73df', linewidth=2)
        ax.set_title('최근 채용 공고 추이', fontsize=16, pad=20)
        ax.set_xlabel('날짜', fontsize=12)
        ax.set_ylabel('채용 공고 수', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        
        # 그래프를 이미지로 저장
        buffer = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        graph_data['trend_graph'] = base64.b64encode(image_png).decode('utf-8')
    
    plt.close(fig)
    
    # 2. 경력별 채용 (Seaborn 사용) - 데이터 전처리 추가
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 불필요한 키워드 및 필터링할 항목들
    irrelevant_keywords = ['홈페이지 지원', '스크랩', '인쇄', '공유', '신고', '미리보기', '지원하기', '마감', '접수양식', '접수방법']
    
    career_skill_data = []
    for detail in JobDetail.objects.all():
        if detail.career_level and detail.skills:
            # 유효하지 않은 경력 데이터 필터링
            if any(keyword in detail.career_level for keyword in irrelevant_keywords):
                continue
            if len(detail.career_level) > 20:  # 비정상적으로 긴 텍스트는 필터링
                continue
                
            # 스킬 수 계산 - 유효한 스킬만 카운트
            skills = re.split(r'[,\s]', detail.skills)
            valid_skills = []
            
            for skill in skills:
                skill = skill.strip().lower()
                # 너무 짧은 단어나 불필요한 키워드 포함된 스킬 필터링
                if len(skill) > 1 and not any(keyword.lower() in skill for keyword in irrelevant_keywords):
                    valid_skills.append(skill)
            
            career_skill_data.append({
                'career': detail.career_level,
                'skill_count': len(valid_skills)
            })
    
    if career_skill_data:
        skill_df = pd.DataFrame(career_skill_data)
        # 경력 레벨별 그룹화
        grouped_df = skill_df.groupby('career')['skill_count'].mean().reset_index()
        # 평균 스킬 수가 0인 경우 제외 (데이터 오류일 가능성 높음)
        grouped_df = grouped_df[grouped_df['skill_count'] > 0]
        # 경력이 비정상적으로 긴 경우 제외
        grouped_df = grouped_df[grouped_df['career'].apply(lambda x: len(str(x)) < 20)]
        # 값 기준으로 내림차순 정렬하고 상위 10개 선택
        grouped_df = grouped_df.sort_values('skill_count', ascending=False).head(10)
        
        sns.barplot(x='skill_count', y='career', data=grouped_df, palette='viridis', ax=ax)
        ax.set_title('경력별 스킬 요구사항 갯수', fontsize=16, pad=20)
        ax.set_xlabel('언급 수', fontsize=12)
        ax.set_ylabel('경력 수준', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3, axis='x')
        
        # 그래프를 이미지로 저장
        buffer = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        graph_data['career_skill_graph'] = base64.b64encode(image_png).decode('utf-8')
    
    plt.close(fig)
    
    return graph_data

def generate_matplotlib_graphs_from_dataframe(df):
    """DataFrame으로부터 Matplotlib 그래프 생성"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
    from matplotlib.ticker import MaxNLocator
    
    plt.switch_backend('Agg')  # 비대화형 백엔드 사용
    
    graph_data = {}
    
    # 1. 생성일자별 채용 공고 추이
    if 'created_at' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        date_counts = df['date'].value_counts().sort_index()
        
        if not date_counts.empty:
            ax.plot(date_counts.index, date_counts.values, marker='o', linestyle='-', color='#4e73df', linewidth=2)
            ax.set_title('채용 공고 추이', fontsize=16, pad=20)
            ax.set_xlabel('날짜', fontsize=12)
            ax.set_ylabel('채용 공고 수', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='x', rotation=45)
            
            # 그래프를 이미지로 저장
            buffer = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            graph_data['trend_graph'] = base64.b64encode(image_png).decode('utf-8')
        
        plt.close(fig)
    
    # 2. 스킬 요구사항과 경력 관계 시각화 (Seaborn 사용)
    if 'career_level' in df.columns and 'skills' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 각 직무에 요구되는 스킬 수 계산
        df['skill_count'] = df['skills'].fillna('').apply(lambda x: len(re.split(r'[,\s]', str(x))))
        
        # 경력별 그룹화하여 평균 스킬 수 계산
        skill_by_career = df.groupby('career_level')['skill_count'].mean().reset_index()
        skill_by_career = skill_by_career.sort_values('skill_count', ascending=False).head(10)
        
        if not skill_by_career.empty:
            sns.barplot(x='skill_count', y='career_level', data=skill_by_career, palette='viridis', ax=ax)
            ax.set_title('경력별 평균 요구 스킬 수', fontsize=16, pad=20)
            ax.set_xlabel('평균 요구 스킬 수', fontsize=12)
            ax.set_ylabel('경력 수준', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3, axis='x')
            
            # 그래프를 이미지로 저장
            buffer = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            graph_data['career_skill_graph'] = base64.b64encode(image_png).decode('utf-8')
        
        plt.close(fig)
    
    return graph_data

def generate_chart_data_from_dataframe(df):
    """백업 파일 DataFrame으로부터 Chart.js 데이터 생성"""
    
    chart_data = {}
    # 직무 분야별 채용 분포
    job_field_data = {}
    if 'job_field' in df.columns:
        field_counter = Counter()
        for field in df['job_field'].dropna():
            fields = re.split(r'[,/]', str(field))
            for f in fields:
                f = f.strip()
                if f and len(f) < 30:
                    field_counter[f] += 1
        
        top_fields = field_counter.most_common(10)
        job_field_data = {
            'labels': [field[0] for field in top_fields],
            'data': [field[1] for field in top_fields]
        }
    chart_data['job_field_data_json'] = json.dumps(job_field_data)
    
    # 지역별 채용 현황
    location_data = {}
    if 'location' in df.columns:
        location_counter = Counter(df['location'].dropna())
        top_locations = location_counter.most_common(10)
        location_data = {
            'labels': [loc[0] for loc in top_locations],
            'data': [loc[1] for loc in top_locations]
        }
    chart_data['location_data_json'] = json.dumps(location_data)
    
    # 경력 수준별 분포
    career_data = {}
    if 'career_level' in df.columns:
        career_counter = Counter(df['career_level'].dropna())
        top_careers = career_counter.most_common(10)
        career_data = {
            'labels': [career[0] for career in top_careers],
            'data': [career[1] for career in top_careers]
        }
    chart_data['career_data_json'] = json.dumps(career_data)
    
    # 학력 요구사항 분포
    education_data = {}
    if 'education_level' in df.columns:
        education_counter = Counter(df['education_level'].dropna())
        top_education = education_counter.most_common(10)
        education_data = {
            'labels': [edu[0] for edu in top_education],
            'data': [edu[1] for edu in top_education]
        }
    chart_data['education_data_json'] = json.dumps(education_data)
    
    # 개발 분야별 채용 비교
    dev_field_comparison = {'labels': ['AI/빅데이터', '웹 개발', '앱 개발'], 'data': [0, 0, 0]}
    if 'job_field' in df.columns:
        ai_jobs = df['job_field'].str.contains('AI|인공지능|머신러닝|딥러닝|빅데이터', case=False, na=False).sum()
        web_jobs = df['job_field'].str.contains('웹|프론트엔드|백엔드|풀스택', case=False, na=False).sum()
        app_jobs = df['job_field'].str.contains('모바일|앱|안드로이드|iOS', case=False, na=False).sum()
        dev_field_comparison['data'] = [ai_jobs, web_jobs, app_jobs]
    chart_data['dev_field_comparison_json'] = json.dumps(dev_field_comparison)
    
    # 기술 스택 분석
    skill_data = {'labels': [], 'data': []}
    if 'skills' in df.columns:
        skill_counter = Counter()
        for skills_str in df['skills'].dropna():
            skills = re.split(r'[,\s]', str(skills_str))
            for skill in skills:
                skill = skill.strip().lower()
                if len(skill) > 1 and len(skill) < 15:
                    skill_counter[skill] += 1
        
        top_skills = skill_counter.most_common(15)
        skill_data = {
            'labels': [skill[0] for skill in top_skills],
            'data': [skill[1] for skill in top_skills]
        }
    chart_data['skill_data_json'] = json.dumps(skill_data)
    
    # 백업 파일에서 생성된 인사이트 (간단한 예시)
    chart_data['trend_insight'] = "백업 파일 데이터로부터 트렌드 인사이트를 생성하였습니다."
    chart_data['location_insight'] = "백업 파일 데이터로부터 위치 인사이트를 생성하였습니다."
    chart_data['career_insight'] = "백업 파일 데이터로부터 경력 인사이트를 생성하였습니다."
    chart_data['education_insight'] = "백업 파일 데이터로부터 교육 인사이트를 생성하였습니다."
    chart_data['skill_insight'] = "백업 파일 데이터로부터 스킬 인사이트를 생성하였습니다."
    
    return chart_data

def ml_insights(request):
    """머신러닝/딥러닝 기반 인사이트 페이지"""
    context = {}
    
    # 모델 디렉토리 확인
    model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 모델 로드 시도 (존재하지 않으면 None 반환)
    try:
        clustering_model = JobClustering.load_model()
        salary_model = SalaryPredictor.load_model()
        trend_model = TrendPredictor.load_model()
        job_classifier = JobFieldClassifier.load_model()
        
        # 모델 상태 확인
        models_ready = {
            'clustering': clustering_model is not None,
            'salary': salary_model is not None,
            'trend': trend_model is not None,
            'classifier': job_classifier is not None
        }
        context['models_ready'] = models_ready
    except Exception as e:
        # 모델 로드 중 오류 발생 시 오류 메시지 표시
        context['model_load_error'] = f"모델 로드 중 오류 발생: {str(e)}"
        context['models_ready'] = {'clustering': False, 'salary': False, 'trend': False, 'classifier': False}
        models_ready = context['models_ready']
    
    # LLM 인사이트 요청 처리
    if request.method == 'POST' and 'generate_insight' in request.POST:
        insight_type = request.POST.get('insight_type')
        data_content = {}
        
        if insight_type == 'trend' and models_ready['trend']:
            # 트렌드 데이터 준비
            date_counts = JobPosting.objects.values('created_at__date').annotate(count=Count('id')).order_by('created_at__date')
            dates = [item['created_at__date'].strftime('%Y-%m-%d') for item in date_counts]
            counts = [item['count'] for item in date_counts]
            
            try:
                # 향후 30일간의 트렌드 예측
                future_trend = trend_model.predict_next_days(
                    [datetime.strptime(d, '%Y-%m-%d').date() for d in dates], 
                    counts, 
                    n_days=30
                )
                data_content = {
                    'current_trend': dict(zip(dates, counts)),
                    'future_trend': {d.strftime('%Y-%m-%d'): round(v, 2) for d, v in future_trend.items()},
                    'avg_expected': round(future_trend.mean(), 2),
                    'max_expected': round(future_trend.max(), 2),
                    'max_date': future_trend.idxmax().strftime('%Y-%m-%d')
                }
                
                # Ollama의 gemma3:1b 모델로 인사이트 생성
                try:
                    prompt = f"""당신은 채용 데이터 분석 전문가입니다. 다음 채용 트렌드 데이터를 분석하여 
                    주요 인사이트와 향후 시장 전망을 설명해주세요. 
                    특히 최고점과 최저점의 의미, 계절적 요인 가능성, 채용 시장에 영향을 미치는 요소들을 고려하세요.
                    
                    데이터: {data_content}
                    
                    1. 주요 트렌드 요약
                    2. 최고점/최저점 분석
                    3. 예상되는 원인
                    4. 구직자를 위한 조언
                    """
                    
                    insight = generate_response_from_ollama(prompt, model="gemma3:1b")
                    context[f'llm_{insight_type}_insight'] = insight
                except Exception as e:
                    context['llm_error'] = f"Ollama API 호출 중 오류: {str(e)}"
            except Exception as e:
                context['trend_error'] = f"트렌드 데이터 준비 중 오류: {str(e)}"
                
        elif insight_type == 'cluster' and models_ready['clustering']:
            # 클러스터링 데이터 준비
            job_descriptions = list(JobDetail.objects.values_list('description', flat=True))
            
            # None 값을 빈 문자열로 대체
            job_descriptions = [desc if desc is not None else "" for desc in job_descriptions]
            
            try:
                if len(job_descriptions) >= 1:  # 1개 이상의 데이터만 있으면 진행
                    labels = clustering_model.predict(job_descriptions)
                    cluster_analysis = clustering_model.analyze_clusters(job_descriptions, labels)
                    data_content = cluster_analysis
                    
                    # Ollama의 gemma3:1b 모델로 인사이트 생성
                    try:
                        prompt = f"""당신은 채용 데이터 분석 전문가입니다. 다음 직무 클러스터링 분석 결과를 해석하여 
                        각 클러스터의 특성과 의미를 설명해주세요.
                        각 클러스터의 주요 키워드가 나타내는 직무 특성과 해당 분야의 채용 시장 상황을 분석하세요.
                        
                        클러스터 데이터: {data_content}
                        
                        1. 각 클러스터 특성 요약
                        2. 클러스터 간 관계 분석
                        3. 가장 활발한 채용 분야 분석
                        4. 관련 직무 추천
                        """
                        
                        insight = generate_response_from_ollama(prompt, model="gemma3:1b")
                        context[f'llm_{insight_type}_insight'] = insight
                    except Exception as e:
                        context['llm_error'] = f"Ollama API 호출 중 오류: {str(e)}"
                else:
                    context['llm_error'] = "클러스터링을 위한 충분한 데이터가 없습니다."
            except Exception as e:
                context['llm_error'] = f"클러스터링 데이터 준비 중 오류: {str(e)}"
                
        elif insight_type == 'salary' and models_ready['salary']:
            # 급여 예측 데이터 준비
            try:
                example_jobs = JobDetail.objects.select_related('job_posting').order_by('?')[:5]
                predictions = []
                
                if example_jobs:
                    for job_detail in example_jobs:
                        job_df = pd.DataFrame({
                            'career_level': [job_detail.career_level],
                            'education_level': [job_detail.education_level],
                            'location': [job_detail.job_posting.location if hasattr(job_detail, 'job_posting') else ''],
                            'skills': [job_detail.skills]
                        })
                        
                        predicted_salary = salary_model.predict(job_df)[0]
                        
                        predictions.append({
                            'title': job_detail.job_posting.job_title if hasattr(job_detail, 'job_posting') else '무제',
                            'career': job_detail.career_level or '미지정',
                            'education': job_detail.education_level or '미지정',
                            'predicted_salary': f"{int(predicted_salary):,}원"
                        })
                    
                    data_content = {
                        'predictions': predictions,
                        'features': salary_model.features[:10] if salary_model.features else []
                    }
                    
                    # Ollama의 gemma3:1b 모델로 인사이트 생성
                    try:
                        prompt = f"""당신은 채용 데이터 분석 전문가입니다. 다음 급여 예측 데이터를 분석하여
                        직무별 급여 차이의 원인과 특징을 설명해주세요.
                        다양한 요소(경력, 학력, 지역 등)가 급여에 미치는 영향과 급여 협상 시 참고할 수 있는 인사이트를 제공하세요.
                        
                        급여 데이터: {data_content}
                        
                        1. 주요 급여 결정 요소 분석
                        2. 직무별 급여 차이 원인
                        3. 동일 직무 내 급여 편차 요인
                        4. 구직자를 위한 급여 협상 조언
                        """
                        
                        insight = generate_response_from_ollama(prompt, model="gemma3:1b")
                        context[f'llm_{insight_type}_insight'] = insight
                    except Exception as e:
                        context['llm_error'] = f"Ollama API 호출 중 오류: {str(e)}"
                else:
                    context['llm_error'] = "급여 예측을 위한 예제 데이터가 없습니다."
            except Exception as e:
                context['llm_error'] = f"급여 데이터 준비 중 오류: {str(e)}"
    
    # 1. 트렌드 예측 (시계열 분석)
    if models_ready['trend']:
        try:
            # 기존 데이터 가져오기
            date_counts = JobPosting.objects.values('created_at__date').annotate(count=Count('id')).order_by('created_at__date')
            dates = [item['created_at__date'] for item in date_counts]
            counts = [item['count'] for item in date_counts]
            
            # 향후 30일간의 트렌드 예측
            future_trend = trend_model.predict_next_days(dates, counts, n_days=30)
            
            # 예측 결과 시각화
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 과거 데이터
            ax.plot(dates, counts, marker='o', linestyle='-', color='blue', linewidth=2, label='실제 데이터')
            
            # 예측 데이터
            ax.plot(future_trend.index, future_trend.values, marker='x', linestyle='--', color='red', linewidth=2, label='예측 데이터')
            
            ax.set_title('향후 30일간 채용 트렌드 예측', fontsize=16, pad=20)
            ax.set_xlabel('날짜', fontsize=12)
            ax.set_ylabel('예상 채용 공고 수', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            
            # 그래프를 이미지로 저장
            buffer = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            context['trend_prediction_graph'] = base64.b64encode(image_png).decode('utf-8')
            plt.close(fig)
            
            # 간단한 요약 통계
            avg_future = future_trend.mean()
            max_future = future_trend.max()
            max_date = future_trend.idxmax().strftime('%Y-%m-%d')
            
            trend_summary = {
                'avg_jobs': round(avg_future, 1),
                'max_jobs': round(max_future, 1),
                'max_date': max_date,
            }
            context['trend_summary'] = trend_summary
            
        except Exception as e:
            context['trend_error'] = f"트렌드 예측 중 오류 발생: {str(e)}"
    
    # 2. 클러스터링 결과 시각화
    if models_ready['clustering']:
        try:
            # 클러스터링을 위한 데이터 가져오기
            job_descriptions = list(JobDetail.objects.values_list('description', flat=True))
            
            # None 값을 빈 문자열로 대체
            job_descriptions = [desc if desc is not None else "" for desc in job_descriptions]
            
            # 데이터가 충분한지 확인 (데이터 개수 제한 완화)
            if len(job_descriptions) >= 1:  # 1개 이상의 데이터만 있으면 진행
                # 클러스터링 수행
                labels = clustering_model.predict(job_descriptions)
                
                # 클러스터 분석
                cluster_analysis = clustering_model.analyze_clusters(job_descriptions, labels)
                context['cluster_analysis'] = cluster_analysis
                
                # 클러스터별 직무 수 시각화
                cluster_counts = Counter(labels)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                clusters = sorted(cluster_counts.keys())
                counts = [cluster_counts[c] for c in clusters]
                
                sns.barplot(x=[f'클러스터 {c+1}' for c in clusters], y=counts, palette='viridis', ax=ax)
                ax.set_title('클러스터별 채용 공고 분포', fontsize=16, pad=20)
                ax.set_xlabel('클러스터', fontsize=12)
                ax.set_ylabel('채용 공고 수', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.3, axis='y')
                
                # 그래프를 이미지로 저장
                buffer = io.BytesIO()
                plt.tight_layout()
                fig.savefig(buffer, format='png')
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                
                context['cluster_graph'] = base64.b64encode(image_png).decode('utf-8')
                plt.close(fig)
            else:
                context['cluster_error'] = "클러스터링을 위한 데이터가 없습니다."
        except Exception as e:
            context['cluster_error'] = f"클러스터링 중 오류 발생: {str(e)}"
    
    # 3. 급여 예측 정보
    if models_ready['salary']:
        try:
            # 급여 예측 모델의 특성 중요도 불러오기
            feature_importance_path = os.path.join(model_dir, 'salary_features.pkl')
            if os.path.exists(feature_importance_path):
                features = joblib.load(feature_importance_path)
                context['salary_features'] = features[:10]  # 상위 10개 특성만 표시
                
                # 예시 직무의 급여 예측
                example_jobs = JobDetail.objects.select_related('job_posting').order_by('?')[:5]
                predictions = []
                
                if example_jobs:
                    for job_detail in example_jobs:
                        # 단일 직무 데이터프레임 생성
                        job_df = pd.DataFrame({
                            'career_level': [job_detail.career_level],
                            'education_level': [job_detail.education_level],
                            'location': [job_detail.job_posting.location if hasattr(job_detail, 'job_posting') else ''],
                            'skills': [job_detail.skills]
                        })
                        
                        # 급여 예측
                        predicted_salary = salary_model.predict(job_df)[0]
                        
                        predictions.append({
                            'title': job_detail.job_posting.job_title if hasattr(job_detail, 'job_posting') else '무제',
                            'career': job_detail.career_level or '미지정',
                            'education': job_detail.education_level or '미지정',
                            'predicted_salary': f"{int(predicted_salary):,}원"
                        })
                    
                    context['salary_predictions'] = predictions
        except Exception as e:
            context['salary_error'] = f"급여 예측 중 오류 발생: {str(e)}"
    
    # LLM 모델이 사용 가능한지 확인
    try:
        llm_model = GemmaModel.get_model()
        context['llm_available'] = True
    except Exception:
        context['llm_available'] = False
    
    return render(request, 'jobapp/ml_insights.html', context)

def train_ml_models(request):
    """머신러닝 모델 학습 페이지"""
    if request.method == 'POST':
        model_type = request.POST.get('model_type', 'all')
        
        try:
            # 학습에 사용할 데이터 준비
            job_postings = JobPosting.objects.all()
            job_details = JobDetail.objects.select_related('job').all()
            
            # 데이터프레임 생성
            posting_data = []
            for posting in job_postings:
                try:
                    detail = posting.details
                    posting_dict = {
                        'id': posting.id,
                        'title': posting.title,
                        'company': posting.company_name,
                        'location': posting.location,
                        'employment_type': posting.employment_type,
                        'created_at': posting.created_at,
                        'career_level': detail.career_level if detail else None,
                        'education_level': detail.education_level if detail else None,
                        'job_field': detail.job_field if detail else None,
                        'description': detail.description if detail else None,
                        'skills': detail.skills if detail else None,
                    }
                    
                    # 급여 정보 추가
                    try:
                        salary = posting.salary
                        posting_dict['salary_min'] = salary.min_salary
                        posting_dict['salary_max'] = salary.max_salary
                    except:
                        posting_dict['salary_min'] = None
                        posting_dict['salary_max'] = None
                        
                    posting_data.append(posting_dict)
                except Exception as e:
                    # 개별 레코드 오류는 무시하고 계속 진행
                    continue
            
            # 데이터프레임 생성
            df = pd.DataFrame(posting_data)
            
            # 백업 데이터로부터 모델 학습이 가능하도록 수정
            if len(df) < 1: # 데이터 갯수 조정
                # 데이터가 없는 경우 백업 파일 사용
                backup_dir = os.path.join(settings.BASE_DIR, 'backup_data')
                csv_files = glob.glob(os.path.join(backup_dir, '*.csv'))
                if csv_files:
                    # 가장 최근 백업 파일 사용
                    latest_file = sorted(csv_files, key=os.path.getmtime, reverse=True)[0]
                    print(f"DB 데이터가 없어 백업 파일을 사용합니다: {latest_file}")
                    df = pd.read_csv(latest_file)
                    messages.info(request, f"DB 데이터가 없어 백업 파일({os.path.basename(latest_file)})을 사용합니다.")
                else:
                    messages.error(request, "모델 학습을 위한 데이터가 없습니다. DB나 백업 파일에 데이터를 추가해주세요.")
                    return redirect('jobapp:train_ml_models')
            
            # 모델 학습
            if model_type == 'all':
                results = train_all_models(df)
                messages.success(request, "모든 머신러닝 모델이 성공적으로 학습되었습니다.")
            elif model_type == 'clustering':
                clustering = JobClustering(n_clusters=5)
                labels = clustering.fit(df['description'].fillna('').tolist())
                results = clustering.analyze_clusters(df['description'].fillna('').tolist(), labels)
                messages.success(request, "클러스터링 모델이 성공적으로 학습되었습니다.")
            elif model_type == 'salary':
                salary_predictor = SalaryPredictor()
                feature_importance = salary_predictor.fit(df)
                results = feature_importance.to_dict('records')
                messages.success(request, "급여 예측 모델이 성공적으로 학습되었습니다.")
            elif model_type == 'trend':
                df['date'] = pd.to_datetime(df['created_at']).dt.date
                date_counts = df['date'].value_counts().sort_index()
                
                trend_predictor = TrendPredictor()
                loss = trend_predictor.fit(date_counts.index.tolist(), date_counts.values.tolist())
                results = {'loss': loss}
                messages.success(request, "트렌드 예측 모델이 성공적으로 학습되었습니다.")
            elif model_type == 'classifier':
                df_clean = df.dropna(subset=['description', 'job_field'])
                
                classifier = JobFieldClassifier()
                feature_importance = classifier.fit(df_clean['description'].tolist(), df_clean['job_field'].tolist())
                results = feature_importance.to_dict('records')
                messages.success(request, "직무 분류 모델이 성공적으로 학습되었습니다.")
            
            return redirect('jobapp:ml_insights')
            
        except Exception as e:
            messages.error(request, f"모델 학습 중 오류가 발생했습니다: {str(e)}")
    
    # GET 요청 처리
    return render(request, 'jobapp/train_ml_models.html')

def generate_response_from_ollama(prompt, model="gemma3:1b"):
    """Ollama API를 사용하여 LLM 응답 생성"""
    try:
        print(f"Ollama API 호출 시작 - 모델: {model}")
        print(f"프롬프트 길이: {len(prompt)} 자")
        
        # Ollama 서버 연결 확인
        try:
            health_check = requests.get("http://localhost:11434/api/tags", timeout=5)
            if health_check.status_code != 200:
                return f"Ollama 서버 연결 오류: 상태 코드 {health_check.status_code}"
        except requests.exceptions.RequestException as e:
            return f"Ollama 서버 연결 실패: {str(e)}. 서버가 실행 중인지 확인하세요."
        
        # 요청 데이터 준비 - 프롬프트 길이 제한
        MAX_PROMPT_LENGTH = 4000  # 안전한 최대 길이
        if len(prompt) > MAX_PROMPT_LENGTH:
            prompt = prompt[:MAX_PROMPT_LENGTH] + "...(잘림)"
            print(f"프롬프트가 너무 길어 {MAX_PROMPT_LENGTH}자로 제한됨")
        
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        # API 요청 - 타임아웃 증가 및 헤더 추가
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=request_data,
            headers=headers,
            timeout=180  # 3분으로 타임아웃 증가 (긴 응답을 위한 시간 확보)
        )
        
        # 응답 확인
        if response.status_code == 200:
            try:
                result = response.json()
                if "response" in result:
                    response_text = result["response"]
                    # 응답 길이 확인
                    print(f"Ollama API 응답 성공: 길이 {len(response_text)}자")
                    return response_text
                else:
                    print(f"Ollama API 응답에 'response' 필드 없음: {result}")
                    return "API 응답에 예상된 필드가 없습니다."
            except json.JSONDecodeError as json_error:
                print(f"JSON 파싱 오류: {str(json_error)}")
                print(f"원본 응답 (일부): {response.text[:200] if response.text else '내용 없음'}")
                return f"API 응답을 JSON으로 파싱할 수 없습니다: {str(json_error)}"
        else:
            error_msg = f"Ollama API 오류 {response.status_code}"
            try:
                error_json = response.json()
                error_msg += f": {error_json.get('error', '상세 정보 없음')}"
            except:
                if response.text:
                    error_msg += f" - {response.text[:200]}"
            
            print(error_msg)
            return f"Error: {error_msg}"
    
    except requests.exceptions.Timeout:
        error_msg = f"Ollama API 타임아웃 (180초 초과)"
        print(error_msg)
        return f"Error: API 요청이 타임아웃되었습니다. 서버가 과부하 상태거나 프롬프트가 너무 복잡할 수 있습니다."
    
    except requests.exceptions.ConnectionError:
        error_msg = "Ollama API 연결 오류 - 서버가 실행 중인지 확인하세요"
        print(error_msg)
        return f"Error: Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."
    
    except Exception as e:
        error_msg = f"Ollama API 호출 중 예외 발생: {str(e)}"
        print(error_msg)
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return f"Error: {str(e)}"

def chat(request):
    """Gemma 3 1B 모델과의 대화 페이지"""
    chat_history = request.session.get('chat_history', [])
    
    if request.method == 'POST':
        if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.content_type == 'application/x-www-form-urlencoded':
            # AJAX 요청 처리
            message = request.POST.get('message', '')
            
            if message:
                try:
                    # 채팅 기록 업데이트 (datetime 객체 대신 문자열 저장)
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    chat_history.append({
                        'content': message,
                        'is_user': True,
                        'timestamp': timestamp
                    })
                    
                    # Ollama의 gemma3:1b 모델로 응답 생성
                    prompt = f"""당신은 채용 데이터 분석 및 취업 상담 전문가입니다. 
                    다음 질문에 전문적이고 유용한 조언을 제공해주세요.
                    채용시장 트렌드, 기술 스택, 경력 관리, 면접 준비 등에 대한 질문에 답변해야 합니다.
                    
                    사용자 질문: {message}
                    """
                    
                    response = generate_response_from_ollama(prompt, model="gemma3:1b")
                    
                    # 채팅 기록 업데이트 (datetime 객체 대신 문자열 저장)
                    chat_history.append({
                        'content': response,
                        'is_user': False,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    # 세션에 채팅 기록 저장 (최대 20개 메시지만 유지)
                    request.session['chat_history'] = chat_history[-20:] if len(chat_history) > 20 else chat_history
                    
                    return JsonResponse({'status': 'success', 'response': response})
                except Exception as e:
                    return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
            else:
                return JsonResponse({'status': 'error', 'message': '메시지가 비어있습니다.'}, status=400)
    
    # GET 요청 처리
    context = {'chat_history': chat_history}
    return render(request, 'jobapp/chat.html', context)

def clear_chat(request):
    """채팅 내역을 초기화하는 함수"""
    if 'chat_history' in request.session:
        del request.session['chat_history']
    return redirect('jobapp:chat')
