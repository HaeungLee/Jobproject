import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import re
import joblib
import os
from django.conf import settings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import requests
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 모델 저장 디렉토리 설정
MODEL_DIR = os.path.join(settings.BASE_DIR, 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)

def clean_text(text):
    """텍스트 전처리 함수"""
    if not isinstance(text, str):
        return ""
    
    # 빈 문자열 체크 추가
    if not text.strip():
        return "빈_설명"  # 기본 텍스트 제공
    
    # 소문자 변환 (영어만 적용)
    text = ''.join([c.lower() if c.isascii() and c.isalpha() else c for c in text])
    
    # 특수 문자는 공백으로 대체하되 한글은 보존
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 결과가 비어있으면 기본값 반환
    return text if text.strip() else "빈_설명"

class JobClustering:
    """채용 공고 클러스터링 모델"""
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        # TF-IDF 벡터화 설정 변경
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # 영어 불용어 제거 비활성화 (한글 텍스트 처리)
            min_df=0.0,        # 최소 문서 빈도를 0으로 설정 (모든 단어 허용)
            max_df=1.0,        # 최대 문서 빈도를 1.0으로 설정 (모든 문서에 나타나도 허용)
            token_pattern=r'(?u)\b\w+\b|[가-힣]+',  # 영어 단어와 한글 포함
            ngram_range=(1, 2)  # 단일 단어와 2단어 조합 모두 사용
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
    def fit(self, job_descriptions):
        """
        채용 공고 설명을 기반으로 클러스터링 모델 학습
        
        Args:
            job_descriptions (list): 채용 공고 설명 텍스트 리스트
        """
        # 텍스트 전처리
        cleaned_descriptions = [clean_text(desc) for desc in job_descriptions]
        
        # TF-IDF 벡터화
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_descriptions)
        
        # KMeans 클러스터링 수행
        self.kmeans.fit(tfidf_matrix)
        
        # 모델 저장
        joblib.dump(self.vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.kmeans, os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
        
        return self.kmeans.labels_
    
    def predict(self, job_descriptions):
        """
        새로운 채용 공고에 대한 클러스터 예측
        
        Args:
            job_descriptions (list): 채용 공고 설명 텍스트 리스트
            
        Returns:
            numpy.ndarray: 클러스터 레이블
        """
        # 텍스트 전처리
        cleaned_descriptions = [clean_text(desc) for desc in job_descriptions]
        
        # TF-IDF 벡터화
        tfidf_matrix = self.vectorizer.transform(cleaned_descriptions)
        
        # 클러스터 예측
        return self.kmeans.predict(tfidf_matrix)
    
    @classmethod
    def load_model(cls):
        """저장된 모델 로드"""
        try:
            vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
            kmeans_path = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
            
            if not os.path.exists(vectorizer_path) or not os.path.exists(kmeans_path):
                print(f"클러스터링 모델 파일이 존재하지 않습니다: {vectorizer_path} 또는 {kmeans_path}")
                return None
            
            vectorizer = joblib.load(vectorizer_path)
            kmeans = joblib.load(kmeans_path)
            
            model = cls(n_clusters=kmeans.n_clusters)
            model.vectorizer = vectorizer
            model.kmeans = kmeans
            
            return model
        except Exception as e:
            print(f"클러스터링 모델 로드 오류: {str(e)}")
            return None
    
    def analyze_clusters(self, job_descriptions, labels):
        """
        클러스터별 주요 키워드 및 특성 분석
        
        Args:
            job_descriptions (list): 채용 공고 설명 텍스트 리스트
            labels (numpy.ndarray): 클러스터 레이블
            
        Returns:
            dict: 클러스터별 분석 결과
        """
        cluster_analysis = {}
        
        # None 값을 빈 문자열로 대체
        job_descriptions = [desc if desc is not None else "" for desc in job_descriptions]
        
        for i in range(self.n_clusters):
            # 현재 클러스터에 속한 직무 설명 텍스트
            cluster_texts = [job_descriptions[j] for j in range(len(job_descriptions)) if labels[j] == i]
            
            # 클러스터가 비어있으면 건너뛰기
            if not cluster_texts:
                continue
            
            # 해당 클러스터의 모든 단어 빈도 계산
            all_words = ' '.join(cluster_texts).lower()
            words = re.findall(r'\b\w+\b', all_words)
            word_freq = Counter(words)
            
            # 불용어 제거 (예: 'the', 'and' 등)
            common_words = {'the', 'and', 'for', 'to', 'in', 'of', 'a', 'with', 'on', 'is', 'as', 'at'}
            for word in common_words:
                if word in word_freq:
                    del word_freq[word]
            
            # 가장 빈번한 단어 10개 추출
            top_words = [word for word, _ in word_freq.most_common(10)]
            
            # 클러스터별 분석 결과 저장
            cluster_analysis[f'클러스터 {i}'] = {
                'top_keywords': top_words,
                'job_count': len(cluster_texts)
            }
        
        return cluster_analysis

class SalaryPredictor:
    """채용 공고 급여 예측 모델"""
    def __init__(self):
        self.model = RandomForestRegressor(random_state=42)
        self.features = None
        
    def prepare_features(self, df):
        """급여 예측을 위한 특성 추출"""
        # 필요한 특성 선택 (이 부분은 실제 데이터에 맞게 수정 필요)
        job_features = df[['career_level', 'education_level', 'location']].copy()
        
        # 범주형 변수에 대한 원-핫 인코딩
        job_features = pd.get_dummies(job_features)
        
        # 스킬 수 특성 추가
        if 'skills' in df.columns:
            job_features['skill_count'] = df['skills'].fillna('').apply(lambda x: len(str(x).split(',')))
        
        # 특성 목록 저장
        self.features = job_features.columns.tolist()
        
        return job_features
    
    def fit(self, df, salary_col='salary_min'):
        """
        모델 학습
        
        Args:
            df (pandas.DataFrame): 채용 데이터 DataFrame (급여 포함)
            salary_col (str): 급여 정보가 있는 열 이름
        """
        # 급여 데이터가 없는 행 제외
        df = df.dropna(subset=[salary_col])
        
        # 급여 값이 숫자가 아닌 경우 처리 (예: "협의", "면접 후 결정" 등)
        df = df[pd.to_numeric(df[salary_col], errors='coerce').notna()]
        df[salary_col] = df[salary_col].astype(float)
        
        # 특성 준비
        X = self.prepare_features(df)
        y = df[salary_col]
        
        # 모델 학습
        self.model.fit(X, y)
        
        # 모델 저장
        joblib.dump(self.model, os.path.join(MODEL_DIR, 'salary_predictor.pkl'))
        joblib.dump(self.features, os.path.join(MODEL_DIR, 'salary_features.pkl'))
        
        # 특성 중요도 계산
        feature_importance = pd.DataFrame(
            {'feature': X.columns, 'importance': self.model.feature_importances_}
        ).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict(self, df):
        """
        급여 예측
        
        Args:
            df (pandas.DataFrame): 채용 데이터 DataFrame
            
        Returns:
            numpy.ndarray: 예측 급여
        """
        # 특성 준비
        X = self.prepare_features(df)
        
        # 모델이 학습된 특성에 맞게 DataFrame 조정
        for feature in self.features:
            if feature not in X.columns:
                X[feature] = 0
        X = X[self.features]
        
        # 급여 예측
        return self.model.predict(X)
    
    @classmethod
    def load_model(cls):
        """저장된 모델 로드"""
        try:
            model = joblib.load(os.path.join(MODEL_DIR, 'salary_predictor.pkl'))
            features = joblib.load(os.path.join(MODEL_DIR, 'salary_features.pkl'))
            
            predictor = cls()
            predictor.model = model
            predictor.features = features
            
            return predictor
        except:
            return None

class TrendPredictor:
    """채용 트렌드 예측 모델 (시계열 분석)"""
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.model = Sequential()
        self.scaler = StandardScaler()
        
    def prepare_data(self, dates, counts):
        """시계열 데이터 준비"""
        # 날짜를 시퀀스로 변환
        date_seq = pd.Series(counts, index=pd.DatetimeIndex(dates)).sort_index()
        
        # 결측일 처리 (0으로 채우기)
        full_date_range = pd.date_range(min(dates), max(dates))
        date_seq = date_seq.reindex(full_date_range, fill_value=0)
        
        # 데이터 스케일링
        scaled_data = self.scaler.fit_transform(date_seq.values.reshape(-1, 1)).flatten()
        
        # 시퀀스 데이터 생성
        X, y = [], []
        for i in range(len(scaled_data) - self.window_size):
            X.append(scaled_data[i:i + self.window_size])
            y.append(scaled_data[i + self.window_size])
        
        return np.array(X), np.array(y), date_seq
    
    def fit(self, dates, counts):
        """
        모델 학습
        
        Args:
            dates (list): 날짜 목록
            counts (list): 각 날짜별 채용 공고 수
        """
        # 시계열 데이터 준비
        X, y, date_seq = self.prepare_data(dates, counts)
        
        # 데이터가 충분한지 확인
        if len(X) < 10:  # 최소 10개 이상의 데이터 포인트 필요
            raise ValueError("시계열 예측을 위한 충분한 데이터가 없습니다.")
        
        # LSTM 모델 구성
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        
        # 모델 컴파일
        self.model.compile(optimizer='adam', loss='mse')
        
        # 모델 학습
        X = X.reshape(X.shape[0], X.shape[1], 1)
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # 모델 저장
        self.model.save(os.path.join(MODEL_DIR, 'trend_predictor.h5'))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'trend_scaler.pkl'))
        joblib.dump(self.window_size, os.path.join(MODEL_DIR, 'trend_window_size.pkl'))
        joblib.dump(date_seq, os.path.join(MODEL_DIR, 'trend_date_seq.pkl'))
        
        return self.model.evaluate(X, y)
    
    def predict_next_days(self, dates, counts, n_days=30):
        """
        향후 n일간의 채용 추세 예측
        
        Args:
            dates (list): 기존 날짜 목록
            counts (list): 각 날짜별 채용 공고 수
            n_days (int): 예측할 미래 일수
            
        Returns:
            pandas.Series: 예측 결과 (날짜 인덱스)
        """
        try:
            # 데이터 유효성 검사 추가
            if len(dates) < self.window_size or len(counts) < self.window_size:
                raise ValueError(f"데이터가 부족합니다. 최소 {self.window_size}개 이상 필요 (현재: {len(dates)}개)")
            
            # 시계열 데이터 준비
            _, _, date_seq = self.prepare_data(dates, counts)
            
            # 마지막 시퀀스 가져오기
            if len(date_seq) < self.window_size:
                raise ValueError(f"처리된 시계열 데이터가 부족합니다. 최소 {self.window_size}개 이상 필요 (현재: {len(date_seq)}개)")
            
            last_sequence = date_seq.values[-self.window_size:].reshape(1, self.window_size, 1)
            
            # 미래 n일 예측
            predictions = []
            curr_sequence = last_sequence.copy()
            
            for _ in range(n_days):
                # 다음 값 예측
                next_pred = self.model.predict(curr_sequence, verbose=0)[0][0]
                predictions.append(next_pred)
                
                # 시퀀스 업데이트 (가장 오래된 값 제거, 새로운 예측 추가)
                curr_sequence = np.append(curr_sequence[:, 1:, :], [[next_pred]], axis=1)
                curr_sequence = curr_sequence.reshape(1, self.window_size, 1)  # 형태 명시적 지정
            
            # 예측값 역스케일링
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            
            # 날짜 범위 생성
            last_date = date_seq.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
            
            # 예측 결과를 Series로 반환
            return pd.Series(predictions, index=future_dates)
        
        except Exception as e:
            print(f"트렌드 예측 오류: {str(e)}")
            # 오류 발생 시 빈 예측 시리즈 반환 (기본값)
            last_date = pd.to_datetime(max(dates) if dates else datetime.now())
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
            return pd.Series([0] * n_days, index=future_dates)
    
    @classmethod
    def load_model(cls):
        """저장된 모델 로드"""
        try:
            model_path = os.path.join(MODEL_DIR, 'trend_predictor.h5')
            scaler_path = os.path.join(MODEL_DIR, 'trend_scaler.pkl')
            window_size_path = os.path.join(MODEL_DIR, 'trend_window_size.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(window_size_path):
                print(f"트렌드 모델 파일이 존재하지 않습니다.")
                return None
            
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            window_size = joblib.load(window_size_path)
            
            predictor = cls(window_size=window_size)
            predictor.model = model
            predictor.scaler = scaler
            
            return predictor
        except Exception as e:
            print(f"트렌드 모델 로드 오류: {str(e)}")
            return None

class JobFieldClassifier:
    """채용 공고 직무 분류 모델"""
    def __init__(self):
        # TF-IDF 벡터화 설정 변경
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,  # 영어 불용어 제거 비활성화 (한글 텍스트 처리)
            min_df=0.0,        # 최소 문서 빈도를 0으로 설정 (모든 단어 허용)
            max_df=1.0,        # 최대 문서 빈도를 1.0으로 설정 (모든 문서에 나타나도 허용)
            token_pattern=r'(?u)\b\w+\b|[가-힣]+',  # 영어 단어와 한글 포함
            ngram_range=(1, 2)  # 단일 단어와 2단어 조합 모두 사용
        )
        self.model = RandomForestClassifier(random_state=42)
        
    def fit(self, job_descriptions, job_fields):
        """
        모델 학습
        
        Args:
            job_descriptions (list): 채용 공고 설명 텍스트 리스트
            job_fields (list): 직무 분야 레이블 리스트
        """
        # 텍스트 전처리
        cleaned_descriptions = [clean_text(desc) for desc in job_descriptions]
        
        # TF-IDF 벡터화
        X = self.vectorizer.fit_transform(cleaned_descriptions)
        y = job_fields
        
        # 모델 학습
        self.model.fit(X, y)
        
        # 모델 저장
        joblib.dump(self.vectorizer, os.path.join(MODEL_DIR, 'field_vectorizer.pkl'))
        joblib.dump(self.model, os.path.join(MODEL_DIR, 'field_classifier.pkl'))
        
        # 특성 중요도 계산
        feature_names = self.vectorizer.get_feature_names_out()
        feature_importance = pd.DataFrame(
            {'feature': feature_names, 'importance': self.model.feature_importances_}
        ).sort_values('importance', ascending=False).head(20)
        
        return feature_importance
    
    def predict(self, job_descriptions):
        """
        직무 분야 예측
        
        Args:
            job_descriptions (list): 채용 공고 설명 텍스트 리스트
            
        Returns:
            numpy.ndarray: 예측된 직무 분야
        """
        # 텍스트 전처리
        cleaned_descriptions = [clean_text(desc) for desc in job_descriptions]
        
        # TF-IDF 벡터화
        X = self.vectorizer.transform(cleaned_descriptions)
        
        # 직무 분야 예측
        return self.model.predict(X)
    
    @classmethod
    def load_model(cls):
        """저장된 모델 로드"""
        try:
            vectorizer = joblib.load(os.path.join(MODEL_DIR, 'field_vectorizer.pkl'))
            model = joblib.load(os.path.join(MODEL_DIR, 'field_classifier.pkl'))
            
            classifier = cls()
            classifier.vectorizer = vectorizer
            classifier.model = model
            
            return classifier
        except:
            return None

class GemmaModel:
    """Gemma LLM 모델을 사용하여 채용 데이터 인사이트 생성"""
    
    def __init__(self, backend="ollama", model_name="gemma3:1b"):
        self.backend = backend  # "huggingface" 또는 "ollama"
        self.model_name = model_name  # Ollama면 "gemma:2b" 형식 사용
        self.is_initialized = False
        # Hugging Face 백엔드를 위한 속성
        self.model = None
        self.tokenizer = None
        """
        Gemma 모델 초기화
        
        Args:
            model_name (str): 사용할 Gemma 모델 이름 
                (예: "google/gemma-2b", "google/gemma-7b" 또는 "google/gemma-7b-it")
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
    def initialize(self):
        """모델 및 토크나이저 로드"""
        if not self.is_initialized:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                self.is_initialized = True
                return True
            except Exception as e:
                print(f"모델 초기화 오류: {str(e)}")
                return False
        return True
    
    def generate_response(self, prompt, max_length=512):
        if self.backend == "huggingface":
            """
            입력 프롬프트에 대한 응답 생성
            
            Args:
                prompt (str): 입력 프롬프트
                max_length (int): 최대 응답 길이
                
            Returns:
                str: 생성된 응답
            """
            if not self.is_initialized:
                success = self.initialize()
                if not success:
                    return "모델을 초기화할 수 없습니다. 서버 로그를 확인하세요."
        elif self.backend == "ollama":
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60
                )
                if response.status_code == 200:
                    return response.json()["response"]
                else:
                    return f"Error: {response.status_code}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
            
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 프롬프트를 제거하고 응답만 반환
            response = response[len(prompt):].strip()
            return response
        
        except Exception as e:
            return f"응답 생성 중 오류 발생: {str(e)}"
            
    def analyze_job_data(self, data_type, data_content):
        """
        채용 데이터에 대한 분석 및 인사이트 생성
        
        Args:
            data_type (str): 분석할 데이터 유형 (예: "trend", "cluster", "salary")
            data_content (dict): 분석 데이터
            
        Returns:
            str: 생성된 분석 결과
        """
        prompts = {
            "trend": f"""
                당신은 채용 데이터 분석 전문가입니다. 다음 채용 트렌드 데이터를 분석하여 
                주요 인사이트와 향후 시장 전망을 설명해주세요. 
                특히 최고점과 최저점의 의미, 계절적 요인 가능성, 채용 시장에 영향을 미치는 요소들을 고려하세요.
                
                데이터: {data_content}
                
                1. 주요 트렌드 요약
                2. 최고점/최저점 분석
                3. 예상되는 원인
                4. 구직자를 위한 조언
            """,
            
            "cluster": f"""
                당신은 채용 데이터 분석 전문가입니다. 다음 직무 클러스터링 분석 결과를 해석하여 
                각 클러스터의 특성과 의미를 설명해주세요.
                각 클러스터의 주요 키워드가 나타내는 직무 특성과 해당 분야의 채용 시장 상황을 분석하세요.
                
                클러스터 데이터: {data_content}
                
                1. 각 클러스터 특성 요약
                2. 클러스터 간 관계 분석
                3. 가장 활발한 채용 분야 분석
                4. 관련 직무 추천
            """,
            
            "salary": f"""
                당신은 채용 데이터 분석 전문가입니다. 다음 급여 예측 데이터를 분석하여
                직무별 급여 차이의 원인과 특징을 설명해주세요.
                다양한 요소(경력, 학력, 지역 등)가 급여에 미치는 영향과 급여 협상 시 참고할 수 있는 인사이트를 제공하세요.
                
                급여 데이터: {data_content}
                
                1. 주요 급여 결정 요소 분석
                2. 직무별 급여 차이 원인
                3. 동일 직무 내 급여 편차 요인
                4. 구직자를 위한 급여 협상 조언
            """
        }
        
        if data_type not in prompts:
            return f"지원되지 않는 데이터 유형: {data_type}"
        
        return self.generate_response(prompts[data_type])
    
    @classmethod
    def get_model(cls, model_name="google/gemma-2b"):
        """
        모델 인스턴스 가져오기 (싱글톤 패턴)
        """
        # 애플리케이션 전역에서 모델 인스턴스 공유를 위한 임시 저장소
        # 실제 운영에서는 Django 캐시나 global_settings 활용 권장
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance
        
# 모델을 미리 초기화하는 함수 (서버 시작 시 실행 가능)
def initialize_llm_model():
    try:
        model = GemmaModel.get_model()
        return model.initialize()
    except Exception as e:
        print(f"LLM 모델 초기화 실패: {str(e)}")
        return False

def train_all_models(df):
    """
    모든 ML 모델 학습
    
    Args:
        df (pandas.DataFrame): 채용 데이터
    
    Returns:
        dict: 각 모델의 학습 결과
    """
    results = {}
    
    # 1. 채용 공고 클러스터링
    if 'description' in df.columns:
        clustering = JobClustering(n_clusters=5)
        labels = clustering.fit(df['description'].fillna('').tolist())
        results['clustering'] = clustering.analyze_clusters(df['description'].fillna('').tolist(), labels)
    
    # 2. 급여 예측
    if 'salary_min' in df.columns:
        try:
            salary_predictor = SalaryPredictor()
            feature_importance = salary_predictor.fit(df)
            results['salary_prediction'] = feature_importance.to_dict('records')
        except Exception as e:
            results['salary_prediction'] = f"급여 예측 모델 학습 실패: {str(e)}"
    
    # 3. 채용 트렌드 예측
    if 'created_at' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['created_at']).dt.date
            date_counts = df['date'].value_counts().sort_index()
            
            trend_predictor = TrendPredictor()
            loss = trend_predictor.fit(date_counts.index.tolist(), date_counts.values.tolist())
            results['trend_prediction'] = {'loss': loss}
        except Exception as e:
            results['trend_prediction'] = f"트렌드 예측 모델 학습 실패: {str(e)}"
    
    # 4. 직무 분류
    if 'description' in df.columns and 'job_field' in df.columns:
        try:
            df_clean = df.dropna(subset=['description', 'job_field'])
            
            classifier = JobFieldClassifier()
            feature_importance = classifier.fit(df_clean['description'].tolist(), df_clean['job_field'].tolist())
            results['field_classification'] = feature_importance.to_dict('records')
        except Exception as e:
            results['field_classification'] = f"직무 분류 모델 학습 실패: {str(e)}"
    
    return results