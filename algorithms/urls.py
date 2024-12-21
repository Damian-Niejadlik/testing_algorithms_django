from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views


urlpatterns = [
    path("", views.home, name="home"),
    path("algorithm/<str:algorithm_name>/", views.algorithm_test, name="algorithm_test"),
    path("start/", views.start_algorithm, name="start_algorithm"),
    path("download/<str:filename>/", views.download_results, name="download_results"),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
