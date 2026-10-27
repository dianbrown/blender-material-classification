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
