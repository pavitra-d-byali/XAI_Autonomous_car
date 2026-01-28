# =========================================================
# FIX_REPO_FINAL.ps1
# Windows-safe permanent dataset removal
# =========================================================

Write-Host "🔥 Removing datasets from Git history (Windows-safe)..." -ForegroundColor Red

# --- Safety check ---
if (!(Test-Path .git)) {
    Write-Host "ERROR: Run this from the git root (folder containing .git)" -ForegroundColor Red
    exit 1
}

# --- Ensure .gitignore ---
$ignores = @(
    ".env",
    "XAI_CAR/data/",
    "data/",
    "venv/",
    "tfenv/",
    "__pycache__/",
    "*.pyc"
)

if (!(Test-Path .gitignore)) {
    New-Item .gitignore -ItemType File | Out-Null
}

foreach ($i in $ignores) {
    if (-not (Select-String .gitignore -Pattern "^$i$" -Quiet)) {
        Add-Content .gitignore $i
    }
}

git add .gitignore
git commit -m "chore: ignore datasets and environment files"

# --- HARD HISTORY REWRITE (THIS IS THE FIX) ---
Write-Host "🚨 Rewriting history to REMOVE datasets..." -ForegroundColor Yellow

python -m git_filter_repo --path XAI_CAR/data --invert-paths --force

# --- Cleanup ---
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# --- Reset remote ---
git remote remove origin 2>$null
git remote add origin https://github.com/pavitra-d-byali/XAI_Autonomous_car.git

# --- Force push clean history ---
Write-Host "🚀 Pushing clean history to GitHub..." -ForegroundColor Cyan
git push origin main --force

Write-Host "✅ DONE. Repo is permanently fixed." -ForegroundColor Green
