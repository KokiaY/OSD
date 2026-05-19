$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$runs = @(
    @{
        Name = "U-Net-10e"
        Config = "config/m4d/baseline_unet_resnet50_10e_bs2.py"
        Output = "fig_results/m4d/baselines_visual_10e/unet"
        CkptDir = "checkpoints/m4d/baseline_unet_resnet50_10e_bs2"
    },
    @{
        Name = "DeepLabV3-10e"
        Config = "config/m4d/baseline_deeplabv3_resnet50_10e_bs2.py"
        Output = "fig_results/m4d/baselines_visual_10e/deeplabv3"
        CkptDir = "checkpoints/m4d/baseline_deeplabv3_resnet50_10e_bs2"
    },
    @{
        Name = "mitb3-SegFormer-10e"
        Config = "config/m4d/baseline_segformer_mitb3_10e_bs2.py"
        Output = "fig_results/m4d/baselines_visual_10e/segformer_mitb3"
        CkptDir = "checkpoints/m4d/baseline_segformer_mitb3_10e_bs2"
    }
)

New-Item -ItemType Directory -Force -Path "train_logs" | Out-Null
New-Item -ItemType Directory -Force -Path "fig_results/m4d/baselines_visual_10e" | Out-Null

foreach ($run in $runs) {
    $safeName = $run.Name -replace "[^A-Za-z0-9]+", "_"
    $trainLog = "train_logs/m4d_baseline_${safeName}_train.log"
    $testLog = "train_logs/m4d_baseline_${safeName}_test.log"
    $lastCkpt = Join-Path $run.CkptDir "last.ckpt"

    if (Test-Path $lastCkpt) {
        Write-Host "===== Skipping training $($run.Name); using $lastCkpt ====="
    }
    else {
        Write-Host "===== Training $($run.Name) ====="
        & python train.py -c $run.Config 2>&1 | Tee-Object -FilePath $trainLog
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $($run.Name). See $trainLog"
        }
        if (-not (Test-Path $lastCkpt)) {
            throw "Missing expected checkpoint after training: $lastCkpt"
        }
    }

    Write-Host "===== Testing $($run.Name) ====="
    & python test_m4d.py -c $run.Config -o $run.Output --rgb -b 1 --workers 1 --ckpt_path $lastCkpt 2>&1 | Tee-Object -FilePath $testLog
    if ($LASTEXITCODE -ne 0) {
        throw "Testing failed for $($run.Name). See $testLog"
    }
}

Write-Host "All 10-epoch M4D baseline runs finished."
