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

## Installer execution and review handoff (15:09 local)

Integrated installer ea18546 from01acfa5. Compiler actualpath in childworktree,
Inno7.1.0; initial fullbuild rejected27untracked.pyc introduced by nativePython
startup, beforecompilation. Nativeworkernow-B8be57b8 plus actualchildflagregression;
19Rusttests/buildpass. Removedonlyuntracked__pycache__/*.pyc with checkedabsolute
packagepaths, allinventoriedpayloadretained. Operationalrefresh script
artifacts/release-v05/refresh_worker_binary.py replacednativeexeandmanifest,
backupsinbefore-worker-startup-fix. FINALexe34c5257c74fcc77ef37b3b6f3bfd0ea0e14ba4897bc6e178dd090deeb44805d2,
12509184bytes. NativeCPUrealpredictionwiththisexecreatedzeroPYC.

Current fullinstaller build SESSION99304 running auto(singlefirst) fromparent.
Preverification22812payloadfilespassed; InnoISCCPID69172 begancompressionaround
15:06 local. Output growing (~509MB at15:09) at
 dist/installer-v0.5/build/single/assets/MaterialStudio-0.5.0-windows-x64-Setup.exe
Do NOTuploadoruseuntilbuildercompletespostverification andmovesfilesinto
 dist/installer-v0.5/release/
Compilerlog dist/installer-v0.5/build/single/compiler.log mayremainbuffereduntilend.
Builderauto repeatsas2,000,000,000byteslicesifEXE>=2GiB. Session99304output will
reportfinish/receipt. Nootherbuild/GPU/UIprocessactive. Failedinitialbuilddidnot
createoutputdirectory, samesuccessfuloutputrootreusedsafely.

IndependentStandards/Spec round1foundonlysameP2 stalehash docs/INSTALLATION.md65.
Corrected98a76cc. Freshrelease05-hash-fix-review-r2confirmedcommand+manifest+EXE
allmatch34c525..., no remainingP1/P2inbothscopes. Summaryreports/RELEASE_V05_REVIEW.md.
Integrated94Python tests pass/4skips(1symlinkpermission+3optincompiler/install;
agentseparatelyranactualtinyfixture37pass1skip),Ruffpass. Initialiconresourcebuild
CPUoptinpassed; startupfixsubprocessfixtureandrealCPUrunpassed. No newscoring.

PR3 draft https://github.com/dianbrown/blender-material-classification/pull/3
linked; branchrelease/v0.5. Willpushdocreview/finalhandoffthenwatchCI toresume.
Notyetmerged/tagged/published. Final installeractualinstall/uninstallQArequired:
freshLocalAppData/MaterialStudio-QA-<guid>; confirmnormalAppIduninstallregistry
notalreadyinuse; optionaluniqueStartMenuGroupwithoutdesktop toverifyshortcuticon.
CLI /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /TASKS="" /DIR=... /GROUP=... .
Verifyinstalledpayloadhashes, nativeexternalcwdCPU/CUDAbridge,actualUIicon/sample,
closeapp; adduser-ownedsentinel thenrunownedunins000.exe silent andproveitsurvives.
Neverrecursive-deletearbitrarydirs. Installer mayaddicon/installreceipt/uninsfiles
outsidepackageinventory; usepackagerverify(whichvalidatesrecordedfiles) not
builderinventory(whichrejectsextras) oninstalledfolder. No-Bdirectruntimecalls
wouldcreatepycache; allnativeworkersnow-B, directtestcallsuse-Btoo.

Finalreleaseassets: setupEXE+(binsifany),SHA256SUMS,buildreceipt,smallMP4optional.
Tests/QA receiptscanbepublishedasseparatevalidationJSON; avoidclaimingbenchmark
numbersfromdemo. Exactv0.5tagdoesnotexistyet. RepositoryPRIVATE remainsunchanged.
NeedfinalPRbodyready/CIpass/mergetomain(userauthorizedthisrelease),createv0.5tag
atfinalmaincommit,uploadassetsandpublishGitHubrelease. NoKagglefullarchiveupload.

## Final installer and clean recording validation, 2026-10-27

The full build completed successfully. The single-file installer exceeded the
GitHub asset limit, so the builder produced a spanned release: Setup.exe,
Setup-1.bin and Setup-2.bin. Only dist/installer-v0.5/release is deliverable.
The final native hash remains 34c5257c74fcc77ef37b3b6f3bfd0ea0e14ba4897bc6e178dd090deeb44805d2.

Actual silent installation into an isolated LocalAppData QA directory passed.
All 22,812 payload files were verified after installation. Native CPU and CUDA
predictions passed from the installed copy. The cube icon, sample and ready
worker were observed in its native window; Start Menu targets were verified.
The app closed gracefully. Silent uninstall returned zero, removed the app,
runtime, shortcuts and registry entry, and preserved a user-owned sentinel.
Evidence: reports/results/demo/release_v05_validation.json and local logs under
artifacts/release-v05. The QA sentinel directory is intentionally retained.

The user rejected the previous recording because it included the Computer Use
cursor overlay. Its replacement is a direct Windows.Graphics.Capture recording
with capture_cursor=0, encoded as an 18-second GIF/MP4. Representative final
frames show the actual prediction, Details, Study and Samples without the
cursor or halo. Browser decode, advancing playback and loop wrap passed.
README now embeds material-studio-loop.gif and links to the release MP4 asset.
Exact media hashes and capture settings: reports/results/demo/showcase_validation.json.

Remaining: push these documentation/media changes, obtain hosted CI on the
final head, merge PR3, tag v0.5 on main, upload and publish the installer files,
checksums, receipt and clean MP4. Keep repository visibility unchanged. No
full Kaggle dataset upload. Unwatch PR3 before the final user handoff.
