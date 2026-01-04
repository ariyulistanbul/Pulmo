# Proje kökünde çalıştığından emin ol
Set-Location $PSScriptRoot

# venv activate
.\venv\Scripts\Activate.ps1

# app'i başlat
python app_offline.py
