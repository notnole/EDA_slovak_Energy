import os, struct

recycle_dir = os.path.join("C:\\$RECYCLE.BIN", "S-1-5-21-258395143-4128789115-3187401415-1002")

for f in os.listdir(recycle_dir):
    if not f.startswith("$I"):
        continue
    fpath = os.path.join(recycle_dir, f)
    try:
        with open(fpath, "rb") as fh:
            data = fh.read()
            if len(data) > 28:
                version = struct.unpack("<Q", data[0:8])[0]
                if version == 2:
                    name = data[28:].decode("utf-16-le", errors="ignore").rstrip("\x00")
                else:
                    name = data[24:].decode("utf-16-le", errors="ignore").rstrip("\x00")
                if "pycharmprojects" in name.lower():
                    rid = "$R" + f[2:]
                    rpath = os.path.join(recycle_dir, rid)
                    exists = "Y" if os.path.exists(rpath) else "N"
                    print(f"{exists}  {name}")
    except:
        pass
