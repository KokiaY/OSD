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

$TtaMode = "d4"
$TestBatchSize = 1
$TestWorkers = 1

$runs = @(
    @{
        Name = "SSP-Net 4cls"
        Config = "config/uavrgb/d2ls_swin_v4_4cls_70e_bs2.py"
        Ckpt = "checkpoints/uavrgb/d2ls_swinv2_base_v4_mpd_4cls_70e_bs2/last.ckpt"
        Output = "fig_results/uavrgb_4cls/sspnet_swinv2_base_v4_mpd_70e_bs2_tta_d4"
        TrainLog = "train_logs/uavrgb_4cls_sspnet_swinv2_base_v4_mpd_70e_bs2_train.log"
        TestLog = "train_logs/uavrgb_4cls_sspnet_swinv2_base_v4_mpd_70e_bs2_test.log"
    },
    @{
        Name = "U-Net 4cls"
        Config = "config/uavrgb/baseline_unet_resnet50_4cls_40e_bs2.py"
        Ckpt = "checkpoints/uavrgb/baseline_unet_resnet50_4cls_40e_bs2/last.ckpt"
        Output = "fig_results/uavrgb_4cls/unet_resnet50_40e_bs2_tta_d4"
        TrainLog = "train_logs/uavrgb_4cls_unet_resnet50_40e_bs2_train.log"
        TestLog = "train_logs/uavrgb_4cls_unet_resnet50_40e_bs2_test.log"
    },
    @{
        Name = "DeepLabV3+ 4cls"
        Config = "config/uavrgb/baseline_deeplabv3plus_resnet50_4cls_40e_bs2.py"
        Ckpt = "checkpoints/uavrgb/baseline_deeplabv3plus_resnet50_4cls_40e_bs2/last.ckpt"
        Output = "fig_results/uavrgb_4cls/deeplabv3plus_resnet50_40e_bs2_tta_d4"
        TrainLog = "train_logs/uavrgb_4cls_deeplabv3plus_resnet50_40e_bs2_train.log"
        TestLog = "train_logs/uavrgb_4cls_deeplabv3plus_resnet50_40e_bs2_test.log"
    },
    @{
        Name = "mitb3-SegFormer 4cls"
        Config = "config/uavrgb/baseline_segformer_mitb3_4cls_40e_bs2.py"
        Ckpt = "checkpoints/uavrgb/baseline_segformer_mitb3_4cls_40e_bs2/last.ckpt"
        Output = "fig_results/uavrgb_4cls/segformer_mitb3_40e_bs2_tta_d4"
        TrainLog = "train_logs/uavrgb_4cls_segformer_mitb3_40e_bs2_train.log"
        TestLog = "train_logs/uavrgb_4cls_segformer_mitb3_40e_bs2_test.log"
    }
)

New-Item -ItemType Directory -Force -Path "train_logs" | Out-Null
New-Item -ItemType Directory -Force -Path "fig_results/uavrgb_4cls" | Out-Null

foreach ($run in $runs) {
    Write-Host ""
    Write-Host "===== Training $($run.Name) ====="
    & $PythonExe train.py -c $run.Config 2>&1 | Tee-Object -FilePath $run.TrainLog
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $($run.Name). See $($run.TrainLog)"
    }

    if (-not (Test-Path -LiteralPath $run.Ckpt)) {
        throw "Missing expected checkpoint after training: $($run.Ckpt)"
    }

    Write-Host ""
    Write-Host "===== Testing $($run.Name) ====="
    $testArgs = @(
        "test_uavrgb.py",
        "-c", $run.Config,
        "-o", $run.Output,
        "--rgb",
        "-b", "$TestBatchSize",
        "--num_workers", "$TestWorkers",
        "--max_visualizations", "0",
        "--ckpt_path", $run.Ckpt
    )
    if ($TtaMode) {
        $testArgs += @("-t", $TtaMode)
    }
    & $PythonExe @testArgs 2>&1 | Tee-Object -FilePath $run.TestLog
    if ($LASTEXITCODE -ne 0) {
        throw "Testing failed for $($run.Name). See $($run.TestLog)"
    }
}

Write-Host ""
Write-Host "All UAVRGB 4-class train/test runs finished."
