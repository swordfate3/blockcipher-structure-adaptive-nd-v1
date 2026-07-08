@echo off
setlocal EnableExtensions EnableDelayedExpansion

set RUN_ID=i1_present_r8_residual_focus_262k_retry1
set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set BRANCH=main
set RUN_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set ARTIFACT_ROOT=%RUN_ROOT%\artifacts
set LOG_DIR=%RUN_ROOT%\logs
set PYTHON_EXE=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%ARTIFACT_ROOT%" mkdir "%ARTIFACT_ROOT%"
echo run_id=%RUN_ID%>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo source_root=%SOURCE_ROOT%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo artifact_root=%ARTIFACT_ROOT%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"

if exist "%SOURCE_ROOT%\.git" (
  cd /d "%SOURCE_ROOT%" || goto failed
  git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1
  git fetch origin %BRANCH% > "%LOG_DIR%\%RUN_ID%_git_fetch_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_git_fetch_stderr.txt" || goto failed
  git checkout %BRANCH% > "%LOG_DIR%\%RUN_ID%_git_checkout_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_git_checkout_stderr.txt" || goto failed
  git pull --ff-only origin %BRANCH% >> "%LOG_DIR%\%RUN_ID%_git_fetch_stdout.txt" 2>> "%LOG_DIR%\%RUN_ID%_git_fetch_stderr.txt" || goto failed
) else (
  if exist "%SOURCE_ROOT%" rmdir /s /q "%SOURCE_ROOT%"
  git clone --branch %BRANCH% "%REPO_URL%" "%SOURCE_ROOT%" > "%LOG_DIR%\%RUN_ID%_git_clone_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_git_clone_stderr.txt" || goto failed
  cd /d "%SOURCE_ROOT%" || goto failed
)

git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>&1
set PYTHONPATH=%SOURCE_ROOT%\src;%PYTHONPATH%
echo pythonpath=%PYTHONPATH%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo started>"%LOG_DIR%\%RUN_ID%_started.marker"

echo command_0>"%LOG_DIR%\%RUN_ID%_command_0.marker"
%PYTHON_EXE% scripts\export-bit-sensitivity-features --eval-plan configs\experiment\innovation1\innovation1_spn_present_r8_trail_position_beamstats_262k_seed0.csv --eval-row-index 1 --split train --feature-view trail_position_stats --dataset-cache-root %ARTIFACT_ROOT%\seed0\dataset_cache\train --progress-output %ARTIFACT_ROOT%\seed0\dataset_cache\seed0_train_feature_export_progress.jsonl --output-dir %ARTIFACT_ROOT%\seed0\train_trail_position_stats_features > "%LOG_DIR%\%RUN_ID%_command_0_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_0_stderr.txt"
if errorlevel 1 goto failed

echo command_1>"%LOG_DIR%\%RUN_ID%_command_1.marker"
%PYTHON_EXE% scripts\export-bit-sensitivity-features --eval-plan configs\experiment\innovation1\innovation1_spn_present_r8_trail_position_beamstats_262k_seed0.csv --eval-row-index 1 --split validation --feature-view trail_position_stats --dataset-cache-root %ARTIFACT_ROOT%\seed0\dataset_cache\validation --progress-output %ARTIFACT_ROOT%\seed0\dataset_cache\seed0_validation_feature_export_progress.jsonl --reference-artifact G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\score_artifacts\trail_position --output-dir %ARTIFACT_ROOT%\seed0\validation_trail_position_stats_features > "%LOG_DIR%\%RUN_ID%_command_1_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_1_stderr.txt"
if errorlevel 1 goto failed

echo command_2>"%LOG_DIR%\%RUN_ID%_command_2.marker"
%PYTHON_EXE% scripts\export-compressed-span-blocks --feature-dir %ARTIFACT_ROOT%\seed0\train_trail_position_stats_features --output-dir %ARTIFACT_ROOT%\seed0\train_span_blocks --output-summary-feature-dir %ARTIFACT_ROOT%\seed0\train_span_summary_features > "%LOG_DIR%\%RUN_ID%_command_2_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_2_stderr.txt"
if errorlevel 1 goto failed

echo command_3>"%LOG_DIR%\%RUN_ID%_command_3.marker"
%PYTHON_EXE% scripts\export-compressed-span-blocks --feature-dir %ARTIFACT_ROOT%\seed0\validation_trail_position_stats_features --output-dir %ARTIFACT_ROOT%\seed0\validation_span_blocks --output-summary-feature-dir %ARTIFACT_ROOT%\seed0\validation_span_summary_features > "%LOG_DIR%\%RUN_ID%_command_3_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_3_stderr.txt"
if errorlevel 1 goto failed

echo command_4>"%LOG_DIR%\%RUN_ID%_command_4.marker"
%PYTHON_EXE% scripts\export-checkpoint-scores --checkpoint G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\checkpoints\row0002_present_trail_position_stats_pairset_seed0.pt --eval-plan configs\experiment\innovation1\innovation1_spn_present_r8_trail_position_beamstats_262k_seed0.csv --eval-row-index 1 --split train --model-key present_trail_position_stats_pairset --hidden-bits 32 --model-options {\"activation\":\"gelu\",\"norm\":\"layernorm\",\"stats_hidden_bits\":64,\"trail_depth\":4,\"trail_words_per_depth\":9} --expert-family trail_position --candidate-status weak_positive --dataset-cache-root %ARTIFACT_ROOT%\seed0\dataset_cache\train_scores --progress-output %ARTIFACT_ROOT%\seed0\dataset_cache\seed0_train_score_export_progress.jsonl --output-dir %ARTIFACT_ROOT%\seed0\train_trail_position_scores > "%LOG_DIR%\%RUN_ID%_command_4_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_4_stderr.txt"
if errorlevel 1 goto failed

echo command_5>"%LOG_DIR%\%RUN_ID%_command_5.marker"
%PYTHON_EXE% scripts\fit-compressed-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed0\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed0\validation_span_summary_features --output-train-dir %ARTIFACT_ROOT%\seed0\train_raw117_scores --output-validation-dir %ARTIFACT_ROOT%\seed0\validation_raw117_scores --output-report %ARTIFACT_ROOT%\seed0\raw117_report.json --run-id i1_present_r8_residual_focus_262k_seed0_raw117 --steps 2000 --learning-rate 0.05 --l2 0.001 --include-feature-prefix aux_depth_cell_ --include-feature-prefix aux_depth_word_ --include-feature-prefix aux_word_global_ --include-feature-prefix primary_depth_trailword_ > "%LOG_DIR%\%RUN_ID%_command_5_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_5_stderr.txt"
if errorlevel 1 goto failed

echo command_6>"%LOG_DIR%\%RUN_ID%_command_6.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed0\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed0\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed0\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed0\residual_focus05_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed0\residual_focus05_validation_scores --output-report %ARTIFACT_ROOT%\seed0\residual_focus05_report.json --run-id i1_present_r8_residual_focus_262k_seed0_focus05 --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --residual-focus-fraction 0.05 --include-feature-prefix aux_depth_word_ --include-feature-prefix aux_word_ > "%LOG_DIR%\%RUN_ID%_command_6_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_6_stderr.txt"
if errorlevel 1 goto failed

echo command_7>"%LOG_DIR%\%RUN_ID%_command_7.marker"
%PYTHON_EXE% scripts\evaluate-residual-slice-correction --train-base-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed0\validation_raw117_scores --validation-corrected-artifact %ARTIFACT_ROOT%\seed0\residual_focus05_validation_scores --focus-fraction 0.05 --output %ARTIFACT_ROOT%\seed0\residual_focus05_slice_eval.json > "%LOG_DIR%\%RUN_ID%_command_7_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_7_stderr.txt"
if errorlevel 1 goto failed

echo command_8>"%LOG_DIR%\%RUN_ID%_command_8.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed0\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed0\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed0\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed0\residual_focus10_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed0\residual_focus10_validation_scores --output-report %ARTIFACT_ROOT%\seed0\residual_focus10_report.json --run-id i1_present_r8_residual_focus_262k_seed0_focus10 --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --residual-focus-fraction 0.1 --include-feature-prefix aux_depth_word_ --include-feature-prefix aux_word_ > "%LOG_DIR%\%RUN_ID%_command_8_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_8_stderr.txt"
if errorlevel 1 goto failed

echo command_9>"%LOG_DIR%\%RUN_ID%_command_9.marker"
%PYTHON_EXE% scripts\evaluate-residual-slice-correction --train-base-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed0\validation_raw117_scores --validation-corrected-artifact %ARTIFACT_ROOT%\seed0\residual_focus10_validation_scores --focus-fraction 0.1 --output %ARTIFACT_ROOT%\seed0\residual_focus10_slice_eval.json > "%LOG_DIR%\%RUN_ID%_command_9_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_9_stderr.txt"
if errorlevel 1 goto failed

echo command_10>"%LOG_DIR%\%RUN_ID%_command_10.marker"
%PYTHON_EXE% scripts\export-bit-sensitivity-features --eval-plan configs\experiment\innovation1\innovation1_spn_present_r8_trail_position_beamstats_262k_seed1.csv --eval-row-index 1 --split train --feature-view trail_position_stats --dataset-cache-root %ARTIFACT_ROOT%\seed1\dataset_cache\train --progress-output %ARTIFACT_ROOT%\seed1\dataset_cache\seed1_train_feature_export_progress.jsonl --output-dir %ARTIFACT_ROOT%\seed1\train_trail_position_stats_features > "%LOG_DIR%\%RUN_ID%_command_10_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_10_stderr.txt"
if errorlevel 1 goto failed

echo command_11>"%LOG_DIR%\%RUN_ID%_command_11.marker"
%PYTHON_EXE% scripts\export-bit-sensitivity-features --eval-plan configs\experiment\innovation1\innovation1_spn_present_r8_trail_position_beamstats_262k_seed1.csv --eval-row-index 1 --split validation --feature-view trail_position_stats --dataset-cache-root %ARTIFACT_ROOT%\seed1\dataset_cache\validation --progress-output %ARTIFACT_ROOT%\seed1\dataset_cache\seed1_validation_feature_export_progress.jsonl --reference-artifact G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\score_artifacts\trail_position --output-dir %ARTIFACT_ROOT%\seed1\validation_trail_position_stats_features > "%LOG_DIR%\%RUN_ID%_command_11_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_11_stderr.txt"
if errorlevel 1 goto failed

echo command_12>"%LOG_DIR%\%RUN_ID%_command_12.marker"
%PYTHON_EXE% scripts\export-compressed-span-blocks --feature-dir %ARTIFACT_ROOT%\seed1\train_trail_position_stats_features --output-dir %ARTIFACT_ROOT%\seed1\train_span_blocks --output-summary-feature-dir %ARTIFACT_ROOT%\seed1\train_span_summary_features > "%LOG_DIR%\%RUN_ID%_command_12_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_12_stderr.txt"
if errorlevel 1 goto failed

echo command_13>"%LOG_DIR%\%RUN_ID%_command_13.marker"
%PYTHON_EXE% scripts\export-compressed-span-blocks --feature-dir %ARTIFACT_ROOT%\seed1\validation_trail_position_stats_features --output-dir %ARTIFACT_ROOT%\seed1\validation_span_blocks --output-summary-feature-dir %ARTIFACT_ROOT%\seed1\validation_span_summary_features > "%LOG_DIR%\%RUN_ID%_command_13_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_13_stderr.txt"
if errorlevel 1 goto failed

echo command_14>"%LOG_DIR%\%RUN_ID%_command_14.marker"
%PYTHON_EXE% scripts\export-checkpoint-scores --checkpoint G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\checkpoints\row0002_present_trail_position_stats_pairset_seed1.pt --eval-plan configs\experiment\innovation1\innovation1_spn_present_r8_trail_position_beamstats_262k_seed1.csv --eval-row-index 1 --split train --model-key present_trail_position_stats_pairset --hidden-bits 32 --model-options {\"activation\":\"gelu\",\"norm\":\"layernorm\",\"stats_hidden_bits\":64,\"trail_depth\":4,\"trail_words_per_depth\":9} --expert-family trail_position --candidate-status weak_positive --dataset-cache-root %ARTIFACT_ROOT%\seed1\dataset_cache\train_scores --progress-output %ARTIFACT_ROOT%\seed1\dataset_cache\seed1_train_score_export_progress.jsonl --output-dir %ARTIFACT_ROOT%\seed1\train_trail_position_scores > "%LOG_DIR%\%RUN_ID%_command_14_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_14_stderr.txt"
if errorlevel 1 goto failed

echo command_15>"%LOG_DIR%\%RUN_ID%_command_15.marker"
%PYTHON_EXE% scripts\fit-compressed-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed1\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed1\validation_span_summary_features --output-train-dir %ARTIFACT_ROOT%\seed1\train_raw117_scores --output-validation-dir %ARTIFACT_ROOT%\seed1\validation_raw117_scores --output-report %ARTIFACT_ROOT%\seed1\raw117_report.json --run-id i1_present_r8_residual_focus_262k_seed1_raw117 --steps 2000 --learning-rate 0.05 --l2 0.001 --include-feature-prefix aux_depth_cell_ --include-feature-prefix aux_depth_word_ --include-feature-prefix aux_word_global_ --include-feature-prefix primary_depth_trailword_ > "%LOG_DIR%\%RUN_ID%_command_15_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_15_stderr.txt"
if errorlevel 1 goto failed

echo command_16>"%LOG_DIR%\%RUN_ID%_command_16.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed1\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed1\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed1\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed1\residual_focus05_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed1\residual_focus05_validation_scores --output-report %ARTIFACT_ROOT%\seed1\residual_focus05_report.json --run-id i1_present_r8_residual_focus_262k_seed1_focus05 --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --residual-focus-fraction 0.05 --include-feature-prefix aux_depth_word_ --include-feature-prefix aux_word_ > "%LOG_DIR%\%RUN_ID%_command_16_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_16_stderr.txt"
if errorlevel 1 goto failed

echo command_17>"%LOG_DIR%\%RUN_ID%_command_17.marker"
%PYTHON_EXE% scripts\evaluate-residual-slice-correction --train-base-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed1\validation_raw117_scores --validation-corrected-artifact %ARTIFACT_ROOT%\seed1\residual_focus05_validation_scores --focus-fraction 0.05 --output %ARTIFACT_ROOT%\seed1\residual_focus05_slice_eval.json > "%LOG_DIR%\%RUN_ID%_command_17_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_17_stderr.txt"
if errorlevel 1 goto failed

echo command_18>"%LOG_DIR%\%RUN_ID%_command_18.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed1\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed1\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed1\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed1\residual_focus10_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed1\residual_focus10_validation_scores --output-report %ARTIFACT_ROOT%\seed1\residual_focus10_report.json --run-id i1_present_r8_residual_focus_262k_seed1_focus10 --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --residual-focus-fraction 0.1 --include-feature-prefix aux_depth_word_ --include-feature-prefix aux_word_ > "%LOG_DIR%\%RUN_ID%_command_18_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_18_stderr.txt"
if errorlevel 1 goto failed

echo command_19>"%LOG_DIR%\%RUN_ID%_command_19.marker"
%PYTHON_EXE% scripts\evaluate-residual-slice-correction --train-base-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed1\validation_raw117_scores --validation-corrected-artifact %ARTIFACT_ROOT%\seed1\residual_focus10_validation_scores --focus-fraction 0.1 --output %ARTIFACT_ROOT%\seed1\residual_focus10_slice_eval.json > "%LOG_DIR%\%RUN_ID%_command_19_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_19_stderr.txt"
if errorlevel 1 goto failed

echo command_20>"%LOG_DIR%\%RUN_ID%_command_20.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed0\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed0\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed0\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed0\residual_uniform_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed0\residual_uniform_validation_scores --output-report %ARTIFACT_ROOT%\seed0\residual_uniform_report.json --run-id i1_present_r8_residual_focus_262k_seed0_uniform --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --include-feature-prefix aux_depth_word_ --include-feature-prefix aux_word_ > "%LOG_DIR%\%RUN_ID%_command_20_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_20_stderr.txt"
if errorlevel 1 goto failed

echo command_21>"%LOG_DIR%\%RUN_ID%_command_21.marker"
%PYTHON_EXE% scripts\evaluate-residual-slice-correction --train-base-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed0\validation_raw117_scores --validation-corrected-artifact %ARTIFACT_ROOT%\seed0\residual_uniform_validation_scores --focus-fraction 0.1 --output %ARTIFACT_ROOT%\seed0\residual_uniform_slice_eval.json > "%LOG_DIR%\%RUN_ID%_command_21_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_21_stderr.txt"
if errorlevel 1 goto failed

echo command_22>"%LOG_DIR%\%RUN_ID%_command_22.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed0\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed0\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed0\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed0\residual_focus10_labelshuffle_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed0\residual_focus10_labelshuffle_validation_scores --output-report %ARTIFACT_ROOT%\seed0\residual_focus10_labelshuffle_report.json --run-id i1_present_r8_residual_focus_262k_seed0_focus10_shuffle --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --residual-focus-fraction 0.1 --include-feature-prefix aux_depth_word_ --include-feature-prefix aux_word_ --shuffle-train-labels --shuffle-seed 9700 > "%LOG_DIR%\%RUN_ID%_command_22_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_22_stderr.txt"
if errorlevel 1 goto failed

echo command_23>"%LOG_DIR%\%RUN_ID%_command_23.marker"
%PYTHON_EXE% scripts\evaluate-residual-slice-correction --train-base-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed0\validation_raw117_scores --validation-corrected-artifact %ARTIFACT_ROOT%\seed0\residual_focus10_labelshuffle_validation_scores --focus-fraction 0.1 --output %ARTIFACT_ROOT%\seed0\residual_focus10_labelshuffle_slice_eval.json > "%LOG_DIR%\%RUN_ID%_command_23_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_23_stderr.txt"
if errorlevel 1 goto failed

echo command_24>"%LOG_DIR%\%RUN_ID%_command_24.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed1\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed1\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed1\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed1\residual_uniform_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed1\residual_uniform_validation_scores --output-report %ARTIFACT_ROOT%\seed1\residual_uniform_report.json --run-id i1_present_r8_residual_focus_262k_seed1_uniform --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --include-feature-prefix aux_depth_word_ --include-feature-prefix aux_word_ > "%LOG_DIR%\%RUN_ID%_command_24_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_24_stderr.txt"
if errorlevel 1 goto failed

echo command_25>"%LOG_DIR%\%RUN_ID%_command_25.marker"
%PYTHON_EXE% scripts\evaluate-residual-slice-correction --train-base-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed1\validation_raw117_scores --validation-corrected-artifact %ARTIFACT_ROOT%\seed1\residual_uniform_validation_scores --focus-fraction 0.1 --output %ARTIFACT_ROOT%\seed1\residual_uniform_slice_eval.json > "%LOG_DIR%\%RUN_ID%_command_25_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_25_stderr.txt"
if errorlevel 1 goto failed

echo command_26>"%LOG_DIR%\%RUN_ID%_command_26.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed1\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed1\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed1\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed1\residual_focus10_labelshuffle_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed1\residual_focus10_labelshuffle_validation_scores --output-report %ARTIFACT_ROOT%\seed1\residual_focus10_labelshuffle_report.json --run-id i1_present_r8_residual_focus_262k_seed1_focus10_shuffle --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --residual-focus-fraction 0.1 --include-feature-prefix aux_depth_word_ --include-feature-prefix aux_word_ --shuffle-train-labels --shuffle-seed 9701 > "%LOG_DIR%\%RUN_ID%_command_26_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_26_stderr.txt"
if errorlevel 1 goto failed

echo command_27>"%LOG_DIR%\%RUN_ID%_command_27.marker"
%PYTHON_EXE% scripts\evaluate-residual-slice-correction --train-base-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed1\validation_raw117_scores --validation-corrected-artifact %ARTIFACT_ROOT%\seed1\residual_focus10_labelshuffle_validation_scores --focus-fraction 0.1 --output %ARTIFACT_ROOT%\seed1\residual_focus10_labelshuffle_slice_eval.json > "%LOG_DIR%\%RUN_ID%_command_27_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_27_stderr.txt"
if errorlevel 1 goto failed

echo command_28>"%LOG_DIR%\%RUN_ID%_command_28.marker"
%PYTHON_EXE% scripts\analyze-residual-bucket-axis-spectrum --feature-dir %ARTIFACT_ROOT%\seed0\train_span_summary_features --bucket-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --bucket-feature logit_gap_abs --bucket-count 5 --top-groups 12 --target residual_loss --output %ARTIFACT_ROOT%\seed0\train_residual_loss_axis_spectrum.json > "%LOG_DIR%\%RUN_ID%_command_28_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_28_stderr.txt"
if errorlevel 1 goto failed

echo command_29>"%LOG_DIR%\%RUN_ID%_command_29.marker"
%PYTHON_EXE% scripts\analyze-residual-bucket-axis-spectrum --feature-dir %ARTIFACT_ROOT%\seed0\train_span_summary_features --bucket-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --bucket-feature logit_gap_abs --bucket-count 5 --top-groups 12 --target residual_error_at_0_5 --output %ARTIFACT_ROOT%\seed0\train_hard_error_axis_spectrum.json > "%LOG_DIR%\%RUN_ID%_command_29_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_29_stderr.txt"
if errorlevel 1 goto failed

echo command_30>"%LOG_DIR%\%RUN_ID%_command_30.marker"
%PYTHON_EXE% scripts\analyze-residual-bucket-axis-spectrum --feature-dir %ARTIFACT_ROOT%\seed1\train_span_summary_features --bucket-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --bucket-feature logit_gap_abs --bucket-count 5 --top-groups 12 --target residual_loss --output %ARTIFACT_ROOT%\seed1\train_residual_loss_axis_spectrum.json > "%LOG_DIR%\%RUN_ID%_command_30_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_30_stderr.txt"
if errorlevel 1 goto failed

echo command_31>"%LOG_DIR%\%RUN_ID%_command_31.marker"
%PYTHON_EXE% scripts\analyze-residual-bucket-axis-spectrum --feature-dir %ARTIFACT_ROOT%\seed1\train_span_summary_features --bucket-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --bucket-feature logit_gap_abs --bucket-count 5 --top-groups 12 --target residual_error_at_0_5 --output %ARTIFACT_ROOT%\seed1\train_hard_error_axis_spectrum.json > "%LOG_DIR%\%RUN_ID%_command_31_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_31_stderr.txt"
if errorlevel 1 goto failed

echo command_32>"%LOG_DIR%\%RUN_ID%_command_32.marker"
%PYTHON_EXE% scripts\summarize-residual-axis-spectrum --spectrum-reports %ARTIFACT_ROOT%\seed0\train_residual_loss_axis_spectrum.json %ARTIFACT_ROOT%\seed0\train_hard_error_axis_spectrum.json %ARTIFACT_ROOT%\seed1\train_residual_loss_axis_spectrum.json %ARTIFACT_ROOT%\seed1\train_hard_error_axis_spectrum.json --min-report-support 2 --output %ARTIFACT_ROOT%\residual_axis_spectrum_summary.json > "%LOG_DIR%\%RUN_ID%_command_32_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_32_stderr.txt"
if errorlevel 1 goto failed

echo command_33>"%LOG_DIR%\%RUN_ID%_command_33.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed0\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed0\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed0\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed0\residual_focus05_source_selected_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed0\residual_focus05_source_selected_validation_scores --output-report %ARTIFACT_ROOT%\seed0\residual_focus05_source_selected_report.json --run-id i1_present_r8_residual_focus_262k_seed0_focus05_source_selected --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --expert-family residual_focus_source_selected_aux --residual-focus-fraction 0.05 --include-feature-prefixes-from-summary %ARTIFACT_ROOT%\residual_axis_spectrum_summary.json > "%LOG_DIR%\%RUN_ID%_command_33_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_33_stderr.txt"
if errorlevel 1 goto failed

echo command_34>"%LOG_DIR%\%RUN_ID%_command_34.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed0\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed0\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed0\train_trail_position_scores %ARTIFACT_ROOT%\seed0\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed0\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed0\residual_focus10_source_selected_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed0\residual_focus10_source_selected_validation_scores --output-report %ARTIFACT_ROOT%\seed0\residual_focus10_source_selected_report.json --run-id i1_present_r8_residual_focus_262k_seed0_focus10_source_selected --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --expert-family residual_focus_source_selected_aux --residual-focus-fraction 0.1 --include-feature-prefixes-from-summary %ARTIFACT_ROOT%\residual_axis_spectrum_summary.json > "%LOG_DIR%\%RUN_ID%_command_34_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_34_stderr.txt"
if errorlevel 1 goto failed

echo command_35>"%LOG_DIR%\%RUN_ID%_command_35.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed1\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed1\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed1\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed1\residual_focus05_source_selected_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed1\residual_focus05_source_selected_validation_scores --output-report %ARTIFACT_ROOT%\seed1\residual_focus05_source_selected_report.json --run-id i1_present_r8_residual_focus_262k_seed1_focus05_source_selected --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --expert-family residual_focus_source_selected_aux --residual-focus-fraction 0.05 --include-feature-prefixes-from-summary %ARTIFACT_ROOT%\residual_axis_spectrum_summary.json > "%LOG_DIR%\%RUN_ID%_command_35_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_35_stderr.txt"
if errorlevel 1 goto failed

echo command_36>"%LOG_DIR%\%RUN_ID%_command_36.marker"
%PYTHON_EXE% scripts\fit-residual-correction-feature-expert --train-feature-dir %ARTIFACT_ROOT%\seed1\train_span_summary_features --validation-feature-dir %ARTIFACT_ROOT%\seed1\validation_span_summary_features --train-base-artifacts %ARTIFACT_ROOT%\seed1\train_trail_position_scores %ARTIFACT_ROOT%\seed1\train_raw117_scores --validation-base-artifacts G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\score_artifacts\trail_position %ARTIFACT_ROOT%\seed1\validation_raw117_scores --output-train-dir %ARTIFACT_ROOT%\seed1\residual_focus10_source_selected_train_scores --output-validation-dir %ARTIFACT_ROOT%\seed1\residual_focus10_source_selected_validation_scores --output-report %ARTIFACT_ROOT%\seed1\residual_focus10_source_selected_report.json --run-id i1_present_r8_residual_focus_262k_seed1_focus10_source_selected --bucket-count 0 --steps 1000 --learning-rate 0.05 --l2 0.001 --residual-focus-background-weight 0.1 --candidate-status residual_focus_262k_candidate --expert-family residual_focus_source_selected_aux --residual-focus-fraction 0.1 --include-feature-prefixes-from-summary %ARTIFACT_ROOT%\residual_axis_spectrum_summary.json > "%LOG_DIR%\%RUN_ID%_command_36_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_command_36_stderr.txt"
if errorlevel 1 goto failed

echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
