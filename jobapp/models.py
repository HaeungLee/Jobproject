from django.db import models

class JobPosting(models.Model):
    # 채용 공고
    company_name = models.CharField(max_length=255)  # 회사명 (추가)
    job_title = models.CharField(max_length=255)     # 구인광고명
    employment_type = models.CharField(max_length=255, blank=True, null=True)  # 고용형태
    recruitment_count = models.IntegerField(blank=True, null=True)  # 모집인원
    location = models.CharField(max_length=255, blank=True, null=True)  # 근무지
    detail_url = models.URLField(blank=True, null=True)  # 상세 페이지 URL
    created_at = models.DateTimeField(auto_now_add=True)  # 데이터 생성 시각

    class Meta:
        unique_together = ('company_name', 'job_title')  # 중복 방지: 회사명+구인광고명 조합
        verbose_name = "Job Posting"
        verbose_name_plural = "Job Postings"

    def __str__(self):
        return f"{self.company_name} - {self.job_title}"

class JobDetail(models.Model):
    # 채용 공고 상세
    job_posting = models.OneToOneField(JobPosting, on_delete=models.CASCADE, related_name='details')  # JobPosting과 관계 설정
    job_field = models.CharField(max_length=255, blank=True, null=True)  # 모집분야
    job_category = models.CharField(max_length=255, blank=True, null=True)  # 직종
    position = models.CharField(max_length=255, blank=True, null=True)        # 직급 
    role = models.CharField(max_length=255, blank=True, null=True)    # 직책 
    description = models.TextField(blank=True, null=True)                 # 상세요강
    responsibilities = models.TextField(blank=True, null=True)            # 주요업무
    skills = models.CharField(max_length=255, blank=True, null=True)  # 스킬
    career_level = models.CharField(max_length=255, blank=True, null=True)  # 경력
    education_level = models.CharField(max_length=255, blank=True, null=True)  # 학력

    def __str__(self):
        return f"Details for {self.job_posting.job_title}"


class CompanyInfo(models.Model):
    #기업 정보(선택적으로 사용)
    company_name = models.CharField(max_length=255, unique=True)  # 회사명 (고유)
    industry = models.CharField(max_length=255, blank=True, null=True)      # 업종
    company_size = models.CharField(max_length=255, blank=True, null=True)  # 기업 규모
    capital = models.CharField(max_length=255, blank=True, null=True)       # 자본금
    employee_count = models.CharField(max_length=255,blank=True, null=True) # 직원 수
    company_take = models.CharField(max_length=255, blank=True, null=True)  # 매출액
    Starting_salary = models.CharField(max_length=255, blank=True, null=True)  # 초임 연봉
    average_salary = models.CharField(max_length=255, blank=True, null=True)  # 전체 평균 연봉
    average_service_year = models.FloatField(blank=True, null=True)           # 평균 근속연수
    establishment_date = models.DateField(blank=True, null=True)  # 설립일
    website = models.URLField(blank=True, null=True)             # 회사 웹사이트

    def __str__(self):
        return self.company_name

class JobSalary(models.Model):
    #연봉 정보(선택적으로 사용)
    job_posting = models.OneToOneField(JobPosting, on_delete=models.CASCADE, related_name='salary')  # JobPosting과 관계 설정
    salary_type = models.CharField(max_length=50, blank=True, null=True)  # 연봉 유형 (연봉, 월급, 시급 등)
    salary_min = models.CharField(max_length=255, null=True)               # 최소 연봉
    salary_max = models.CharField(max_length=255, null=True)               # 최대 연봉
    working_hours = models.CharField(max_length=255, blank=True, null=True)  # 근무시간
    work_schedule = models.CharField(max_length=255, blank=True, null=True)  # 근무일정(주5일, 주6일 등)

    def __str__(self):
        return f"Salary for {self.job_posting.job_title}"


class ApplicantStats(models.Model):
   #지원자 통계 (선택적으로 사용)
    job_posting = models.ForeignKey(JobPosting, on_delete=models.CASCADE)  # 관련 채용 공고
    age_group = models.CharField(max_length=50, blank=True, null=True)     # 연령대
    gender = models.CharField(max_length=20, blank=True, null=True)        # 성별
    applicant_count = models.IntegerField(default=0)                       # 지원자 수
    education = models.CharField(max_length=100, blank=True, null=True)   # 학력
    

    def __str__(self):
        return f"Stats for {self.job_posting.job_title}"