import os
import glob
CHUNK_SIZE = 50 * 1024 * 1024 # 50 MB
for pkl in glob.glob("models/rf_*.pkl"):
    print(f"Splitting {pkl}...")
    with open(pkl, 'rb') as f:
        chunk = 0
        while True:
            data = f.read(CHUNK_SIZE)
            if not data: break
            out_name = f"{pkl}.part{chunk:02d}"
            with open(out_name, 'wb') as out:
                out.write(data)
            chunk += 1
print("Done splitting.")
