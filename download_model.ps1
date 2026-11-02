# Reproduce the fixed model snapshot without an online inference API.
param([string]$Endpoint = 'https://hf-mirror.com')
$ErrorActionPreference = 'Stop'
$revision = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
$targetDir = Join-Path $PSScriptRoot 'models/bge-small-en-v1.5'
New-Item -ItemType Directory -Path (Join-Path $targetDir '1_Pooling') -Force | Out-Null
$modelFiles = @('1_Pooling/config.json','README.md','config.json','config_sentence_transformers.json','model.safetensors','modules.json','sentence_bert_config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt')
foreach ($modelFile in $modelFiles) {
    $modelUrl = "$Endpoint/BAAI/bge-small-en-v1.5/resolve/$revision/$modelFile"
    curl.exe --silent --show-error -L --fail --retry 3 --max-time 600 $modelUrl -o (Join-Path $targetDir $modelFile)
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $modelFile" }
}
$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $targetDir 'model.safetensors')).Hash.ToLowerInvariant()
if ($actualHash -ne '3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad') { throw 'Model SHA256 mismatch' }
$revision | Set-Content -LiteralPath (Join-Path $targetDir 'REVISION')
Write-Output "Verified model: $targetDir"
