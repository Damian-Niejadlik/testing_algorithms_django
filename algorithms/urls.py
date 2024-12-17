from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('algorithm/<str:algorithm_name>/', views.algorithm_test, name='algorithm_test'),
    path('algorithm/<str:algorithm_name>/', views.algorithm_test, name='algorithm_test'),
    path('algorithm/<str:algorithm_name>/start/', views.start_algorithm, name='start_algorithm'),
    path('algorithm/<str:algorithm_name>/progress/', views.algorithm_progress, name='algorithm_progress'),
]
