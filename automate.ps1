# Automation script for Stock Price Prediction Project

Write-Host "--- Step 1: Installing dependencies ---" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "--- Step 2: Running the prediction analysis ---" -ForegroundColor Cyan
python main.py

Write-Host "--- Step 3: Preparing for GitHub ---" -ForegroundColor Cyan
if (!(Test-Path .git)) {
    git init
    git branch -M main
    Write-Host "Git repository initialized." -ForegroundColor Green
}

git add .
git commit -m "Initial commit: Stock price prediction with LSTM"

$remoteUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git) or press Enter to skip pushing"

if ($remoteUrl -ne "") {
    if (git remote | Select-String "origin") {
        git remote set-url origin $remoteUrl
    } else {
        git remote add origin $remoteUrl
    }
    
    Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
    git push -u origin main
    Write-Host "Push complete!" -ForegroundColor Green
} else {
    Write-Host "Skipping push. You can push manually later using 'git push origin main'." -ForegroundColor Yellow
}
