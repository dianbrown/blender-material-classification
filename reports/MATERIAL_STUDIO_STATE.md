# Material Studio orchestration state

Current branch `feat/material-studio`, base main64adc07 (PR #1 merged at user's
explicit request; former local/remote v3/material-classification removed).
User explicitly confirmed native Rust GPUI after linked private reference repo
proved to contain Python exercises/static HTML rather than a GPUI app.
Group-member names and student numbers must be left blank, per user.

## Active tasks

All T3-owned children use codex/gpt-6-astra, high reasoning/default service,
isolated worktrees from558707e; no duplicate cargo or GPU training jobs.

- studio-gpui-app-20261027-r1: app/** excluding app/assets/study.json and samples;
  scripts/demo/build_app.ps1 if needed. Sole cargo build owner, jobs4, official
  Zed GPUI revision aed37547d8543000494fb69101c05daec5891068. Worktree t3-7684f8fb.
- studio-inference-worker-20261027-r1: src/material_classifier/demo/**,
  scripts/demo/{worker,prepare_assets}.py, tests/test_demo*.py except package,
  app/assets/study.json +21 sample renders. May do short actual inference smoke,
  no retraining or fresh benchmark reassessment. Parent told no GPU jobs active.
- studio-academic-report-20261027-r1: scripts/report/build_project_report.py,
  reports/project/**, reports/ASSIGNMENT_COVERAGE.md, tests/test_project_report.py.
  New final report follows ITRI62610–15body-page recommended structure. Preserve
  old reports internally, no V1/version/Windows/ZIP/agent operational narration.
- studio-presentation-20261027-r1: reports/presentation/**,
  scripts/report/build_presentation.*, docs/CLASSROOM_DEMO.md. Editable slides,
  PDF and speaking notes with blank author fields, all-slide rendered QA.
- studio-portable-package-20261027-r1: scripts/demo/package_app.py,
  tests/test_demo_package.py, docs/DEMO_PACKAGING.md. Portable-folder runtime,
  exact existing model/environment identities, no full datasets/cached training
  features. No GPU or cargo. Final build waits integrated worker/app/assets.

Task IDs use prefix
`node:delegated-task:command%3Amcp%3A432140c9-e9ff-453b-a0c6-4b2557810a8f%3Adelegate-task%3A`
plus name above. Child thread IDs replace node: with thread:.

## Integration

Worker JSONL contract in MATERIAL_STUDIO_PLAN.md. Runtime root contains
app/assets/study.json and samples with paths relative app/assets. Models keep
portable relative artifacts/v3 locations. Native app supports --resources ROOT
or MATERIAL_STUDIO_ROOT, --python PATH or MATERIAL_STUDIO_PYTHON; default bundled
ROOT/runtime/python.exe then repository .venv. No personal absolute paths in code.
Portable folder target dist/MaterialStudio/MaterialStudio.exe plus resources and
runtime. Final package must be tested from outside source checkout.

Backend static schema accepted/forwarded frontend:
schema_version1; class_names array; primary_model frozen_selected;
default_model dinov2_finetuned; results_label/live_label; models array of id,
display_name,role,family,accuracy,macro_f1, accuracy_ci/macro_f1_ci objects
{low,high,level,unit}, confusion_matrix, per_class array with class/P/R/F1/support,
latency {median_ms,p95_ms,device,scope,batch_size,repetitions}, size
{artifact_bytes,total_deployment_bytes}, relative source references.
dataset {development:{images,materials,source_groups},holdout:{...},
classes:[{name,definition,development_images,holdout_images}],scope}.
samples [{id,label,class_name,path,source_group,material_id,split:development,
attribution,source_url,license}], optionalstoredpredictions omitted unless truly
compatible. 21 images, three distinct development groups/class, private PNG
metadata stripped without altering pixels. Live inference never faked by OOF.

## Tools and validation

Parent read complete computer-use skill, guidance and confirmations; initialized
@oai/sky in mcp__node_repl__js. Parent owns native UI interaction once executable
ready. Use sky list_apps/list_windows to select returned exactwindow, then
get_window_state/action/refresh. Never parallel mouse control or guessed handles.
No PowerShell UIAutomation mixed with computer-use. Parent available to test
actual Classify/results/gallery/filepicker, responsive load/error and close.
Python baseC:/Python313, venv parent/.venv, Torch CUDA directory4.09GiB,
~248GBdiskfree initially. Existing pretrained cache under user .cache/huggingface.

## Tracking and remaining parent work

Linear project f3dbf884-bb97-4ace-90ae-141d5ba9ae23 reopened InProgress,
solelead/assignee HectoLogic50bc7339-0c0c-49f6-83a5-1c97ff8a3f2e.
PER374nativeapp;PER375worker/package;PER376report/presentation;PER377review/docs.
Durable document742e8b58-e455-46f0-9dd2-8b1d7f651cec updated.
Parent must integrate commits, updatepublicREADME/modelcard asneeded, build/run
packagedexe andnativeUIQA, measuredstartup/inference, independent code/specreview,
report/PPTpageQA, relevant tests/CI,newfeaturePR and registerwithT3.
Kaggle publication remains deferred to owner. Preserve existing locked study
results and scientific endpoint distinctions while changing presentation.

Early assets82d9edf integratedparent10923b2; 2assettests pass. Frontendreceived
commit/schemaandcanbuildwithactual21samples. NativeGPUIrequiresRust1.98.1;
installed1.99.0selectedbyapp-localtoolchain withoutchangingdefault1.94.
PackagerrequiresnativeLICENSE/NOTICEbundle; appagentaskedtosupply actualCargo
dependency inventory/texts. Runtimepreparationunderartifacts/material-studio.
Workerownerannouncedshortsequential4modelcold/warmCUDAthenCPUsmokeonone
developmentgalleryimage; nootherGPUjobsuntilrelease. ParentdoesnoGPUwork.
DraftPR#2 https://github.com/dianbrown/blender-material-classification/pull/2
createdandT3linked, feat/material-studio->main. NoPRwatchstarted. PER377linked.

Worker live smoke16/16requests passed:fourmodels,cold+warmCUDAthenCPU onone
developmentwoodsample, validorderedprobabilities,exactrepeatperdevice,cleanJSONL
andshutdown. Ready84.7msCUDA/88.5msCPU. WarmCUDAms:DINO20.1,ResNet15.8,
selected34.9,control14.7; CPU107.5/39.6/193.5/44.9. Demonstrationtimingsonly,
notreplacementsforstudybenchmarks. GPUreleasedafterfootprintprobe:allfourcached
503676928bytesallocated,559939584reserved; processworking set1519013888bytes.
ControlsharesConvNeXtcache. WorkerfinishingexactreviewedloaderCPUcomparison.

Additionalchild studio-native-notices-20261027-r1 ownsapp/THIRD_PARTY_NOTICES.txt
andapp/licenses/** (cededbyUIagent), optionalcollector script. Itreadsactual
appCargo.lock/cache; noCargobuild. AppagentaskedtoproduceWindowsfilteredoffline
metadataJSON atidleboundarytoartifacts/material-studio/native-metadata.json.
Packagerwillrequirethatnoticebundlebeforefullassembly. No other GPU jobs active.

## Latest continuation: monochrome redesign

Integrated workerb77bef7, bundledHFoverride18c79da, report e35468c,
presentationbcccb92+format98f5365+recommendation1e9fbe4, notices010ffdb+
format aa80650, native2e1470b+reviewfix8c2661d, packager1ede7ea.
FullPython suite566passed2Windowspermission skips1legacywarning,220.16sec;
Ruffpassesallnewdemo/reportcode. Native15tests+realCPUvalid-invalid-validpass;
structuredimageerrorspreserveworker/cache. Independent fixreviewfoundno newP1/P2;
slide11recommendationresolved; UIvisualfixphysicalQAinitiallypending.

NativeUIactuallytestedafteruserunlockeddesktop: maximize triggeredproperrender;
Classify,Study/Results,Dataset,Aboutviewsrendered; liveDINOwood95.2%,metal91.9%,
file-picker-selectedfabric93.7% andConvNeXt fabric95.2% observed. CtrlO/Enter/
Escape andCtrl1 worked; sample/modelchangesclearedoldresults; navigationduring
firstloadresponsive. Captionfixesnowvisible. NativefilepickerUIAfocused_element
incorrectlyreportedSearch; visualkeyboardfocus+ShiftTabthenfilenameentry worked.
Saved QA screenshots artifacts/material-studio/ui-qa. Do notcommitfiledialog
screenshots showing unrelated personal directories. UserphysicalEscape stopped
ComputerUseonce; userexplicitlyresumedafterasyncrequest. SkyAPIavailable via
mcp__node_repl__js @oai/sky; guidance/api/confirmationsread. GlobalstudioWindow
mustalwaysrefreshedfromreturnedlist_windows; stalehandlecaptureonceincorrectly
showedT3foreground, noinputwasperformedthere. Do notautomateT3UI.

UserthenexplicitlyREJECTEDteal/panel-heavyUI, providedpublicreference
https://github.com/hardbeat920/monocode andscreenshotattachment
C:/Users/dianb/.t3/userdata/attachments/bd9a1f79-755b-456a-8521-821ea121f9c6-80f5e464-e1f1-472a-b1ce-c9758c639fa3.png.
Currentactivefollowup studio-monochrome-redesign-20261027-r1 onoriginalappchild:
160pxcharcoalsidebar,13pxneutraltype,subtleacrylic,NOgradients/colors/glow/cardhero;
singleimage,compactbottomtoolbar(Open/Samples/modelmenu/Classify),inlineresult,
optionalDetails. SecondaryStudy4rowtable+grayscalematrix;Samplesgrid/filter;
Aboutexplanations. UserconfirmedgenuineRustGPUIearlier; keepframework.
OldwindowPID60708closedviafreshSky48172826AltF4; appauthorauthorizedrebuildonce
exits. NoGPUworkactive. NoCargo/dependencychangesexpected=>noticesunchanged.

Packagingcompleted dist/MaterialStudio(candidateoldexe) and
dist/MaterialStudio-final(correctedoldtealstyle,nowALSOcandidateafterredesign),
finaloldexeSHA cd139a6cadc5f6f130cf6406a3f2dd741409a40e81a6e0b0c151ef6177218487.
Each~4.8GiB; reusableartifacts/material-studio/package-runtime4.602GiB verified.
Secondbuildsession21885 completedexit0,files22797,payload5147247344bytes.
Do notdeliveroldstylepackages. FinalnewUIrequiresnewpackagingoutputfolder or
safeverifiednativeupdate; packagerrefusesexistingfolder. Preparedruntimeavoids
recopyingvenv. Fullcommand in docs/DEMO_PACKAGING.md, requiresnativeDLL
C:/Windows/System32/vcruntime140.dll plusnoticebundle. /dist/nowignored.

READMEproductdraft exists artifacts/material-studio/README-draft.md; NOTyetcopied
toREADME. Fixpresentationlink material_classification.pptx,finaldistpath,newnav
Study/Samples/Detailsandusernewminimalstylebeforepublishing. UpdateMODEL_CARD/
DATASET reportlinkstonewreports/project/report.pdf AFTERpackaginginputssettled.
NativeCIworkflowaddedcc273b6: Windowsreleasepassed10m23s onoldhead;PythonCI
failedcollectorlintthenfixedaa80650 locally. Lastpushcc273b6;latercommitsnotyet
pushed. Needfinalpush/tests/PR2body/readyafterredesign, noPRwatchactive.
Newpresentationfollowup studio-presentation-minimal-nav askedauthoralignnav/
demoDetailsreferenceswithnewUIonly, preservemetrics andQAchangedpages.

## Recovered continuation and final design

Recovered completed monochrome implementation afd695a after the child usage-limit
failure; integrated as bbc42d4. Presentation navigation commit 942401b integrated
as 8a4ee8c. README/model/data/coverage update 62a86f2 integrated as 57e66b0.
No implementation was lost, and no training or study rescoring occurred.

Actual native QA completed on monochrome views. Live leather 92.2% and wood
95.2% DINO results observed; Details, model changes, popovers, Study matrix,
Samples filtering, About, keyboard classification and graceful close verified.
Found busy-label overlap and corrected nonshrinking labels/spacer sizing.
Independent new review found hidden disclosures consuming Escape. Fixed in
5a27bc2 with current-page/prediction visibility gates and regression; independent
follow-up reports resolved and no remaining scoped P1/P2. 19 Rust tests pass,
plus successful release build. Final executable SHA256:
65531935d8c785358f3d85c5382f37b3a970b716e6be5b8a7daf4ee868bacbdb.

New package building at dist/MaterialStudio-release via session 73055. Never
use the old teal MaterialStudio or MaterialStudio-final candidates as delivery.
Prepared runtime reused; copied sources/DATASET must remain unchanged during
assembly. All UI processes closed. Operational final worker smoke script at
artifacts/material-studio/verify_final_package.py runs all four models cold/warm
on CPU then CUDA from external temp cwd, bundled runtime, conflicting HF vars.
Run only after package assembly. Then launch packaged exe from external cwd
without resources/python overrides and verify actual native prediction.

Documentation screenshot docs/images/material-studio.png is actual monochrome
native app. Final README/coverage currently honestly mark final package QA
pending; replace those statements after actual completion. Review document
updated. Remaining: package smoke/native launch, final evidence/docs status,
push PR2 and CI, finish Linear. Feature PR2 remains draft; prior PR1 already
merged/main updated and v3 branch deleted. No Kaggle/public binary upload.

## Final portable verification and requested Acrylic adjustment

The monochrome portable package completed: 22,798 files, 5,151,660,569 bytes.
All four models passed cold/warm CPU and CUDA smoke (16 predictions), empty
stderr, exact repeated probabilities, external cwd, no PYTHONPATH, system-only
PATH and conflicting host cache vars. Packaged native app launched without
resource/Python overrides and completed live DINO wood95.2% prediction.
User physical Escape stopped the close/capture action; no further UI input was
made. User then requested slightly more translucency like Zen. UI-only commit
515b650 reduces root alpha232->176, sidebar184->56, toolbar237->200; foreground
and opaque mode unchanged. Official Zen Acrylic docs checked. Release build
passes; new exe SHA25635fac15c99305572a1a33baf55c2acd1ce495ef5fab155b685861019cb4e2cc2.
The local package refresh verifies all prior payload hashes, saves original
binary/manifest and updates only exe+manifest; operational script
artifacts/material-studio/refresh_glass_binary.py, session93010. No new UI driving
requested/performed for this opacity-only pass. Need collect completion then
packaged Rust-worker smoke, final docs/commit/push/CI. Model paths and Kaggle
state remain unchanged. Focused report/demo/package suite67passed36.71sec;
native notice validation311deps/445texts passes with two known exceptions.

Final package refresh completed, all22,797 prior payload hashes verified; only
native exe and package manifest updated, backups retained. Current packaged
native Rust-to-Python smoke from externalcwd passed CUDA wood95.2296%, startup
120.87ms, firstload16.25sec; runtimeandweightsunchanged. Evidence
artifacts/material-studio/glass-native-smoke.json and glass-refresh.json.
All local implementation/report/package deliverables complete; final GitHub CI
is next. No further desktop automation performed after the physicalEscape stop.
