# PowerShell 脚本
$stagedFiles = git diff --cached --name-only
$report = @()

foreach ($file in $stagedFiles) {
    $size = (Get-Item $file).Length / 1MB
    $report += [PSCustomObject]@{
        File = $file
        Size_MB = [math]::Round($size, 2)
    }
}

$report | Sort-Object Size_MB -Descending | Format-Table -AutoSize