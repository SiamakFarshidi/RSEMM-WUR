# app/urls.py
from django.urls import path
from .views import landing_page, estimate_maturity,list_stored_zenodo_records,load_stored_record

app_name = "RSEMM"



urlpatterns = [
    path('', landing_page, name='landing_page'),
    path('estimate/', estimate_maturity, name='estimate_maturity'),
    path('list_stored_records/', list_stored_zenodo_records, name='list_stored_zenodo_records'),
    path('load_stored_record/<str:filename>/', load_stored_record, name='load_stored_record'),

]