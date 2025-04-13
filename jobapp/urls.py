from django.urls import path
from . import views

app_name = 'jobapp'

urlpatterns = [
    path('', views.home, name='home'),
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/<int:job_id>/', views.job_detail, name='job_detail'),
    path('insights/', views.job_insights, name='job_insights'),
    path('ml-insights/', views.ml_insights, name='ml_insights'),
    path('train-models/', views.train_ml_models, name='train_ml_models'),
    path('chat/', views.chat, name='chat'),
    path('clear-chat/', views.clear_chat, name='clear_chat'),
]