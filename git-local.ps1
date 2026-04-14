param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GitArgs
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoName = Split-Path $repoRoot -Leaf
$defaultGitDir = Join-Path $env:USERPROFILE ".codex-git\$repoName.git"
$gitDir = if ($env:D2LS_GIT_DIR) { $env:D2LS_GIT_DIR } else { $defaultGitDir }

git --git-dir="$gitDir" --work-tree="$repoRoot" @GitArgs
