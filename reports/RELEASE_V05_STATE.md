# v0.5 release continuation

Branch release/v0.5, base main ca9efdf. User explicitly authorizes creating and
publishing GitHub release tag v0.5, Windows installer, cube icon/favicon and README
loop. Repo currently PRIVATE; do not change visibility. No full Kaggle ZIP upload.
Linear PER-378 uuid07574ec6-07ba-436b-951b-76502dd840fb, soleHectoLogic.

Integrated commits:5e237d9plan,c90eb65icon,7308b81showcase,96ba8b6notices,
d3bb90dversionedpackager. Current uncommitted READMErelease links andCIinstallerlint.
Iconagent81bb8f complete;MSVC14.44.35207(link14.44.35228),nativeversion0.5.0,
all9iconresourcesverified,GPUImanifestunchanged;19Rusttests+realCPUtestpassed.
Final nativeexe shared artifacts/material-studio/cargo-target/release/material-studio.exe
SHA579a53ba797cb874ddafbf7f71aa4891a9be73c320c4981f90381710f8f1dc2e.
Icon app/assets/icon.ico SHA387b3582cac51259f84fcbf7f870d56b3f5a8a20da004143d62d51e95f39bcc1.
FreshCargo metadata artifacts/material-studio/native-metadata-v05.json.

Noticeagent07faa01 complete.311deps445texts12upstreamlicensefilesvalidated,
2MITnoticegapsremainexplicit(seahash/simd_helpers). NVIDIA CUDA12.8/cuDNN9.19
andMicrosoftconditionaltermsretainedinapp/licenses/upstream. No certification or
inventedlicense/entitlement. Use actualinstalledVSREDISTnativeDLL:
C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC143.CRT/vcruntime140.dll
SHA d5e4d9a3e835fa679450145d6a7d94e36573a509317111904d9b3712c30d9066.

Freshpackage COMPLETE dist/MaterialStudio-v0.5,22812payloadentries,
5147500218bytes. Manifest6887a577a373098add0dbc69c368282cb30dd3bcfcff57ad5ee523d348c79e36.
Built using preparedruntime, finalexe, newnotices/REDIST, --release-version0.5.0.
Fullpayload/sourceidentitychecks+CPUimportsmoke passed. NativeCPU/CUDAbridge
smoke independentlypassed fromtemp cwd; artifacts/release-v05/native-cpu.json
andnative-cuda.json. No modeltraining/study scoring. NoGUI/GPUprocessactive.

ActualGUIshowcasecapturedusingSkybeforeicon-onlyresourceupdate; samefinalUI.
8frames artifacts/release-v05/showcase. GIFdocs/images/material-studio-demo.gif:
178frames,17.8sec,960x646,loop0,1050517bytes; MP4300912bytesalso committed.
Windowtitlebarcropped,pausesshortened,actualwood95.2%/metal91.9% predictions.
Allsourceframes+encodedrepresentativesinspected. Manifest reports/results/demo/showcase_validation.json.
FFmpeg isolateduvtool path C:/Users/dianb/AppData/Local/uv/cache/archive-v0/E9yRxOTM-UfkZb5Y/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe
(notbundled). Skycontrol via mcp__node_repl__js; appclosedAltF4without interruption.

Activechild release05-installer-20261027-r1 (T3 task/node standard prefix),
worktree C:/Users/dianb/.t3/worktrees/blender-material-classification/t3-7d26daf1.
Ownsinstaller/**,scripts/demo/build_installer.py,tests/test_installer.py,
docs/INSTALLATION.md. Sourceimplemented,36tests+1symlinkskip passed including
realTinyInno single/spannedcompile. Finishing tinyinstall/uninstallpreservecheck
anddocs; ETA5mins stated14:52. No largecompile yet. Tool:
C:/Users/dianb/.t3/worktrees/blender-material-classification/t3-7d26daf1/artifacts/installer-tools/Inno Setup 7/ISCC.exe
7.1.0 signaturevalidatedPyrsys. Builderrequires--source--output--icon
--exe-sha256--icon-sha256--compiler, --layoutauto(default). Singlecompressedfirst,
fallback2000000000byteslices if>=2GiBGitHublimit. Outputrelease/receipt/checksums.
FilebasenameMaterialStudio-0.5.0-windows-x64-Setup.issrecipe uses peruser
LocalAppData/Programs/MaterialStudio, x64os, MinWindows10.0.18362, normal
StartMenushortcutsoptionaldesktop, no recursiveUninstallDelete. Parenthasnot
modifiedchildfiles. Returncommitnotification pending. Needintegratecommit,run
fullbuilder withexactimmutablepackage; independentreviewwhilecompressionruns.

Remaining: actuallargeinstallerbuild, safeisolatedinstall(/VERYSILENT /NOICONS
/DIR=artifacts/release-v05/install-test), verifyallinstalledmanifestfiles/native
CPU+CUDAactualprediction andicon, uninstallpreservesuseraddedfile. Freshparallel
Standards/Spec review beforepublish. READMElinkdocs/INSTALLATION currentlypending
childfile. MakePR/review/CI/merge asappropriate(userauthorizedrelease), tagv0.5
atfinalsourcecommit, ghreleaseuploadinstallerassets(each<2GiB),checksums/video,
publishrelease. Do NOT claimfinishuntilactualreleaseexists. ToolwatchPRonlywhen
readytowait; unwatchforhandoff. MainpriorPR1/2merged,noremainingopenPRs initially.
