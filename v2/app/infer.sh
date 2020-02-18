
#!/usr/bin/env bash
python /app/generate_data.py 'test' '/test/' 'test_all.csv.gz'
python /app/infer.py
