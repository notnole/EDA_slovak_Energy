import os, struct, shutil, datetime

recycle_dir = os.path.join("C:\\$RECYCLE.BIN", "S-1-5-21-258395143-4128789115-3187401415-1002")
restore_base = r"C:\Users\noelp\pycharmprojects\ipesoft_eda_data"

print("Recycle dir exists:", os.path.exists(recycle_dir))

ipesoft_files = []
all_today = []

for f in os.listdir(recycle_dir):
    if not f.startswith("$I"):
        continue
    fpath = os.path.join(recycle_dir, f)
    try:
        with open(fpath, "rb") as fh:
            data = fh.read()
            if len(data) > 28:
                version = struct.unpack("<Q", data[0:8])[0]
                file_size = struct.unpack("<Q", data[8:16])[0]
                timestamp = struct.unpack("<Q", data[16:24])[0]
                epoch_diff = 116444736000000000
                ts_seconds = (timestamp - epoch_diff) / 10000000
                dt = datetime.datetime.fromtimestamp(ts_seconds)

                if version == 2:
                    name = data[28:].decode("utf-16-le", errors="ignore").rstrip("\x00")
                else:
                    name = data[24:].decode("utf-16-le", errors="ignore").rstrip("\x00")

                rid = "$R" + f[2:]

                if dt.date() == datetime.date(2026, 4, 9):
                    all_today.append((dt, name, rid, file_size, f))

                if "ipesoft" in name.lower() or "eda_data" in name.lower():
                    ipesoft_files.append((name, rid, file_size, f))
    except Exception as e:
        pass

print(f"\nFiles deleted today: {len(all_today)}")
print(f"Found {len(ipesoft_files)} ipesoft_eda_data entries\n")

print("=== IPESOFT FILES ===")
for name, rid, size, ifile in sorted(ipesoft_files):
    rpath = os.path.join(recycle_dir, rid)
    is_dir = os.path.isdir(rpath)
    print(f"{'DIR' if is_dir else 'FILE':>4}  {size:>12}  {name}")
    print(f"      Recycle ID: {rid}")
    if is_dir:
        for root, dirs, files in os.walk(rpath):
            for fn in files[:10]:
                rel = os.path.relpath(os.path.join(root, fn), rpath)
                print(f"        - {rel}")
            if len(files) > 10:
                print(f"        ... and {len(files)-10} more files")
    print()

print("\n=== ALL FILES DELETED TODAY ===")
for dt, name, rid, size, f in sorted(all_today):
    print(f"{dt.strftime('%H:%M:%S')}  {size:>12}  {name}")
