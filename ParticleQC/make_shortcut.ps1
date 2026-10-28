$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath('Desktop')
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path $desktop '입자 품질검사 ParticleQC.lnk'))
$lnk.TargetPath = Join-Path $dir '2_실행.bat'
$lnk.WorkingDirectory = $dir
$lnk.IconLocation = (Join-Path $dir 'icon.ico') + ',0'
$lnk.WindowStyle = 7
$lnk.Description = 'ParticleQC 범용 입자 AI 품질검사 · NEXT CELL'
$lnk.Save()
Write-Host '  바탕화면에 "입자 품질검사 ParticleQC" 아이콘을 만들었습니다.'
