nohup /home/yin/.venv/bin/jupyter nbconvert \
  --to notebook --execute --allow-errors \
  nodag/nodag_gs_test_lam.ipynb \
  --output nodag/nodag_gs_test_lam_run.ipynb > run.log 2>&1 &

tail -f run.log

