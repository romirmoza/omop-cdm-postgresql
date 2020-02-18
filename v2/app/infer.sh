
#!/usr/bin/env bash
python /app/generate_data.py 'test' '/infer/' 'test_all.csv.gz'
python /app/infer.py
