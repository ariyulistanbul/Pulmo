# run_train.ps1
$ErrorActionPreference = "Stop"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

# ---- venv activate ----
if (-Not (Test-Path ".\venv\Scripts\Activate.ps1")) {
  throw "venv bulunamadi. Once .\setup_venv.ps1 calistir."
}
.\venv\Scripts\Activate.ps1

# ---- dataset root txt ----
$DATASET_TXT = Join-Path $ROOT "dataset_root.txt"
if (-Not (Test-Path $DATASET_TXT)) {
  throw "dataset_root.txt yok. Ornek: D:\dataset\lidc-idr (tek satir)."
}

$DATASET_ROOT = (Get-Content $DATASET_TXT -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($DATASET_ROOT)) {
  throw "dataset_root.txt bos."
}
if (-Not (Test-Path $DATASET_ROOT)) {
  throw "dataset_root.txt icindeki yol bulunamadi: $DATASET_ROOT"
}

# ---- find manifest-*/LIDC-IDRI ----
$MANIFEST_ROOT = $null
$lidc = Get-ChildItem -Path $DATASET_ROOT -Directory -ErrorAction Stop |
  Where-Object { $_.Name -like "manifest-*" } |
  ForEach-Object {
    $cand = Join-Path $_.FullName "LIDC-IDRI"
    if (Test-Path $cand) { $cand } else { $null }
  } | Where-Object { $_ -ne $null } | Select-Object -First 1

if ($null -eq $lidc) {
  throw "manifest-*/LIDC-IDRI bulunamadi. dataset_root.txt icindeki root'u kontrol et."
}
$MANIFEST_ROOT = $lidc

# ---- reader csv: önce proje klasörü, yoksa dataset root ----
function Find-ReaderCsv($baseDir) {
  $preferred = @(
    (Join-Path $baseDir "lidc_reader_level.csv"),
    (Join-Path $baseDir "lidc_reader_level_vs.csv"),
    (Join-Path $baseDir "lidc_reader_level_v5.csv")
  )
  foreach ($p in $preferred) { if (Test-Path $p) { return $p } }

  $cand = Get-ChildItem -Path $baseDir -Filter "*.csv" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "reader" -and $_.Name -match "level" } |
    Select-Object -First 1
  if ($cand) { return $cand.FullName }
  return $null
}

$READER_CSV = Find-ReaderCsv $ROOT
if ($null -eq $READER_CSV) { $READER_CSV = Find-ReaderCsv $DATASET_ROOT }

if ($null -eq $READER_CSV) {
  throw "reader_level CSV bulunamadi. Projede ya da dataset_root icinde lidc_reader_level*.csv olmali."
}

# ---- TRAIN AYARLARI ----
$BACKBONE    = "resnet18"
$EPOCHS      = 12
$BATCH_SIZE  = 32
$K           = 1
$NEG_RATIO   = 1.0
$LR          = 1e-4

$CACHE_DIR = Join-Path $ROOT "cache"
$OUT_DIR   = Join-Path $ROOT "outputs"

Write-Host "[i] dataset_root:  $DATASET_ROOT"  -ForegroundColor Cyan
Write-Host "[i] manifest_root: $MANIFEST_ROOT" -ForegroundColor Cyan
Write-Host "[i] reader_csv:    $READER_CSV"    -ForegroundColor Cyan
Write-Host "[i] out_dir:       $OUT_DIR"       -ForegroundColor Cyan
Write-Host "[i] cache_dir:     $CACHE_DIR"     -ForegroundColor Cyan

python train.py `
  --manifest_root "$MANIFEST_ROOT" `
  --reader_csv "$READER_CSV" `
  --backbone "$BACKBONE" `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --k $K `
  --neg_ratio $NEG_RATIO `
  --lr $LR `
  --cache_dir "$CACHE_DIR" `
  --out_dir "$OUT_DIR"
