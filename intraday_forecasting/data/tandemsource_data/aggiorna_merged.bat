@echo off
setlocal
python "%~dp0merge_csv.py"
if errorlevel 1 (
    echo.
    echo Aggiornamento non completato. Leggere l'errore riportato sopra.
) else (
    echo.
    echo Cartella merged aggiornata.
)
pause
