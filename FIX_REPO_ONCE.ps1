# =========================================================
# FIX_REPO_ONCE.ps1
# PERMANENTLY remove datasets from Git history
# Run ONCE. This rewrites history.
# =========================================================

Write-Host "⚠️  FIXING REPO: REMOVING DATASETS FROM HISTORY" -ForegroundColor Red

# ---------- Safety check ----------
if (!(Test-Path .git)) {
    Write-Host "ERROR: Not a git repository" -ForegroundColor Red
    exit 1
}

# ---------- Ensure git-filter-repo ----------
git filter-repo --help > $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: git-filter-repo not installed" -ForegroundColor Red
    exit 1
}

# ---------- Ensure ignores ----------
$ignore = @(
    ".env",
    "data/",
    "venv/",
    "tfenv/",
    "__pycache__/",
    "*.pyc"
)

if (!(Test-Path .gitignore)) {
    New-Item .gitignore -ItemType File | Out-Null
}

foreach ($i in $ignore) {
    if (-not (Select-String .gitignore -Pattern "^$i$" -Quiet)) {
        Add-Content .gitignore $i
    }
}

git add .gitignore
git commit -m "chore: ignore datasets and environment files" -q

# ---------- HARD RESET HISTORY ----------
Write-Host "🔥 Rewriting history (this is intentional)..." -ForegroundColor Yellow

git filter-repo --path data/ --invert-paths --force

# ---------- Clean leftovers ----------
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# ---------- Restore remote ----------
git remote remove origin 2>$null
git remote add origin https://github.com/pavitra-d-byali/XAI_Autonomous_car.git

# ---------- Force push clean history ----------
Write-Host "🚀 Pushing clean history to GitHub..." -ForegroundColor Cyan
git push origin main --force

Write-Host "✅ DONE. Dataset permanently removed. Repo is clean." -ForegroundColor Green
