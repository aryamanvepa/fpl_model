@echo off
cd /d D:\fpl
"C:\Users\ADMIN\anaconda3\python.exe" -u -m fpl_bot.scripts.train_model1 > D:\fpl\fpl_bot\data\train_model1.log 2>&1
echo DONE_MARKER >> D:\fpl\fpl_bot\data\train_model1.log
