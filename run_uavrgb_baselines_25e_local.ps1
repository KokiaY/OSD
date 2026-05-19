$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$PythonExe = $env:PYTHON_EXE
if (-not $PythonExe) {
    $PythonExe = "E:\MyApps\Anaconda3_2025\envs\d2ls\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$runs = @(
    @{
        Name = "U-Net"
        Config = "config/uavrgb/baseline_unet_resnet50_25e_bs2.py"
        ResumeConfig = "config/uavrgb/baseline_unet_resnet50_25e_bs2_resume.py"
        Output = "fig_results/uavrgb/baselines_visual_25e/unet"
        CkptDir = "checkpoints/uavrgb/baseline_unet_resnet50_25e_bs2"
    },
    @{
        Name = "DeepLabV3+"
        Config = "config/uavrgb/baseline_deeplabv3plus_resnet50_25e_bs2.py"
        ResumeConfig = "config/uavrgb/baseline_deeplabv3plus_resnet50_25e_bs2_resume.py"
        Output = "fig_results/uavrgb/baselines_visual_25e/deeplabv3plus"
        CkptDir = "checkpoints/uavrgb/baseline_deeplabv3plus_resnet50_25e_bs2"
    },
    @{
        Name = "mitb3-SegFormer"
        Config = "config/uavrgb/baseline_segformer_mitb3_25e_bs2.py"
        ResumeConfig = "config/uavrgb/baseline_segformer_mitb3_25e_bs2_resume.py"
        Output = "fig_results/uavrgb/baselines_visual_25e/segformer_mitb3"
        CkptDir = "checkpoints/uavrgb/baseline_segformer_mitb3_25e_bs2"
    }
)

New-Item -ItemType Directory -Force -Path "train_logs" | Out-Null
New-Item -ItemType Directory -Force -Path "fig_results/uavrgb/baselines_visual_25e" | Out-Null

foreach ($run in $runs) {
    $safeName = $run.Name -replace "[^A-Za-z0-9]+", "_"
    $trainLog = "train_logs/uavrgb_baseline_${safeName}_25e_bs2_train.log"
    $testLog = "train_logs/uavrgb_baseline_${safeName}_25e_bs2_test.log"
    $lastCkpt = Join-Path $run.CkptDir "last.ckpt"
    $trainConfig = $run.Config

    if (Test-Path $lastCkpt) {
        $trainConfig = $run.ResumeConfig
        Write-Host "===== Resuming training $($run.Name) from $lastCkpt ====="
    }
    else {
        Write-Host "===== Training $($run.Name) from scratch ====="
    }

    & $PythonExe train.py -c $trainConfig 2>&1 | Tee-Object -FilePath $trainLog
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $($run.Name). See $trainLog"
    }
    if (-not (Test-Path $lastCkpt)) {
        throw "Missing expected checkpoint after training: $lastCkpt"
    }

    Write-Host "===== Testing $($run.Name) ====="
    & $PythonExe test_uavrgb.py -c $run.Config -o $run.Output --rgb -t d4 -b 1 --num_workers 1 --ckpt_path $lastCkpt 2>&1 | Tee-Object -FilePath $testLog
    if ($LASTEXITCODE -ne 0) {
        throw "Testing failed for $($run.Name). See $testLog"
    }
}

Write-Host "All 25-epoch UAVRGB baseline runs finished."
