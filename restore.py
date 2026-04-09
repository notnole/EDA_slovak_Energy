import os, struct, shutil

recycle_dir = os.path.join("C:\\$RECYCLE.BIN", "S-1-5-21-258395143-4128789115-3187401415-1002")

ipesoft_files = []

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
                if version == 2:
                    name = data[28:].decode("utf-16-le", errors="ignore").rstrip("\x00")
                else:
                    name = data[24:].decode("utf-16-le", errors="ignore").rstrip("\x00")

                rid = "$R" + f[2:]
                if "ipesoft" in name.lower() or "eda_data" in name.lower():
                    ipesoft_files.append((name, rid, file_size, f))
    except Exception as e:
        pass

print(f"Found {len(ipesoft_files)} entries to restore:\n")

for orig_path, rid, size, ifile in ipesoft_files:
    r_path = os.path.join(recycle_dir, rid)
    is_dir = os.path.isdir(r_path)
    print(f"{'DIR' if is_dir else 'FILE'}  {orig_path}")
    print(f"  -> Source: {r_path}")
    print(f"  -> Dest:   {orig_path}")

    # Create parent directory if needed
    parent = os.path.dirname(orig_path)
    os.makedirs(parent, exist_ok=True)

    if os.path.exists(orig_path):
        print(f"  !! SKIPPED - destination already exists")
        continue

    try:
        if is_dir:
            shutil.copytree(r_path, orig_path)
            file_count = sum(len(files) for _, _, files in os.walk(orig_path))
            print(f"  OK - restored directory ({file_count} files)")
        else:
            shutil.copy2(r_path, orig_path)
            print(f"  OK - restored file ({size} bytes)")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

# Verify
print("\n=== Verification ===")
base = r"C:\Users\noelp\PycharmProjects\Ipesoft_EDA_data"
if os.path.exists(base):
    for item in os.listdir(base):
        full = os.path.join(base, item)
        if os.path.isdir(full):
            count = sum(len(files) for _, _, files in os.walk(full))
            print(f"  DIR  {item}/ ({count} files)")
        else:
            print(f"  FILE {item} ({os.path.getsize(full)} bytes)")
else:
    print(f"  Base dir does not exist: {base}")
