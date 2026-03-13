@echo off
REM ==========================================
REM PCA Probe Visualization Runner
REM ==========================================

set PYTHON=python

set BASE_HS=base_hs_train.npy
set METHOD_HS=PB_J_hs_train.npy
set LABELS=wmdp_tf_pairs_train.csv
set SCRIPT=pca_probe_viz_band_report.py

set LAYER1=12
set LAYER2=22

set COMP=40

set OUTDIR=comp_pca_layers%LAYER1%_to_%LAYER2%_pca%COMP%

echo.
echo ==========================================
echo Running PCA probe analysis
echo Layer: %LAYER%
echo ==========================================
echo.

%PYTHON% %SCRIPT% ^
 --base_hs %BASE_HS% ^
 --post_hs %METHOD_HS% ^
 --tf_pairs_csv %LABELS% ^
 --split train ^
 --layers %LAYER1%-%LAYER2% ^
 --band_mode concat ^
 --out_dir %OUTDIR%  ^
 --pca_components %COMP%

echo.
echo ==========================================
echo Finished
echo Output folder: %OUTDIR%
echo ==========================================
pause