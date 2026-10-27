# Prerequisites: Rust 1.99.0 (app/rust-toolchain.toml), Visual Studio C++ Build
# Tools (MSVC x64), and a Windows 10/11 SDK including fxc.exe. Official GPUI
# discovers fxc in the SDK registry; set GPUI_FXC_PATH to override that location.
# Release builds embed the shaders; the shader compiler is not a runtime dependency.
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [string]$TargetDir = '',
    [switch]$SkipTests
)
$ErrorActionPreference = 'Stop'
if (-not $TargetDir) { $TargetDir = Join-Path $RepoRoot 'artifacts/material-studio/cargo-target' }
$previousTarget = $env:CARGO_TARGET_DIR
try {
    $env:CARGO_TARGET_DIR = [IO.Path]::GetFullPath($TargetDir)
    Push-Location (Join-Path $RepoRoot 'app')
    # The crate pins the installed stable toolchain required by the official GPUI revision.
    & cargo fmt --all -- --check
    if ($LASTEXITCODE -ne 0) { throw 'Rust formatting check failed.' }
    & cargo check --locked -j 4
    if ($LASTEXITCODE -ne 0) { throw 'Rust type check failed.' }
    if (-not $SkipTests) {
        & cargo test --locked --release -j 4
        if ($LASTEXITCODE -ne 0) { throw 'Rust tests failed.' }
    }
    & cargo build --locked --release -j 4
    if ($LASTEXITCODE -ne 0) { throw 'Release build failed.' }
    Write-Output (Join-Path $env:CARGO_TARGET_DIR 'release/material-studio.exe')
} finally {
    Pop-Location
    $env:CARGO_TARGET_DIR = $previousTarget
}
