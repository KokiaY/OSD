$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$runs = @(
    @{
        Name = "U-Net"
        Config = "config/m4d/baseline_unet_resnet50_60e_bs2.py"
        Output = "fig_results/m4d/baselines_visual/unet"
        CkptDir = "checkpoints/m4d/baseline_unet_resnet50_60e_bs2"
        CkptPattern = "baseline_unet_resnet50_60e_bs2*.ckpt"
    },
    @{
        Name = "DeepLabV3"
        Config = "config/m4d/baseline_deeplabv3_resnet50_60e_bs2.py"
        Output = "fig_results/m4d/baselines_visual/deeplabv3"
        CkptDir = "checkpoints/m4d/baseline_deeplabv3_resnet50_60e_bs2"
        CkptPattern = "baseline_deeplabv3_resnet50_60e_bs2*.ckpt"
    },
    @{
        Name = "mitb3-SegFormer"
        Config = "config/m4d/baseline_segformer_mitb3_60e_bs2.py"
        Output = "fig_results/m4d/baselines_visual/segformer_mitb3"
        CkptDir = "checkpoints/m4d/baseline_segformer_mitb3_60e_bs2"
        CkptPattern = "baseline_segformer_mitb3_60e_bs2*.ckpt"
    }
)

New-Item -ItemType Directory -Force -Path "train_logs" | Out-Null
New-Item -ItemType Directory -Force -Path "fig_results/m4d/baselines_visual" | Out-Null

foreach ($run in $runs) {
    $safeName = $run.Name -replace "[^A-Za-z0-9]+", "_"
    $trainLog = "train_logs/m4d_baseline_${safeName}_60e_bs2_train.log"
    $testLog = "train_logs/m4d_baseline_${safeName}_60e_bs2_test.log"

    $existingCkpt = Get-ChildItem -Path $run.CkptDir -Filter $run.CkptPattern -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($existingCkpt) {
        Write-Host "===== Skipping training $($run.Name); using $($existingCkpt.FullName) ====="
    }
    else {
        Write-Host "===== Training $($run.Name) ====="
        & python train.py -c $run.Config 2>&1 | Tee-Object -FilePath $trainLog
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $($run.Name). See $trainLog"
        }
        $existingCkpt = Get-ChildItem -Path $run.CkptDir -Filter $run.CkptPattern |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
    }

    Write-Host "===== Testing $($run.Name) ====="
    & python test_m4d.py -c $run.Config -o $run.Output --rgb -b 1 --workers 1 --ckpt_path $existingCkpt.FullName 2>&1 | Tee-Object -FilePath $testLog
    if ($LASTEXITCODE -ne 0) {
        throw "Testing failed for $($run.Name). See $testLog"
    }
}

Write-Host "All M4D baseline runs finished."
