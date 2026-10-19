#!/usr/bin/env python3
"""Size and 16 KB page alignment of the native libraries in an APK.

Android 15+ devices with 16 KB pages need every native library's LOAD
segments aligned to 16 KB, and (stored uncompressed, as this app stores
them) its offset in the APK aligned to 16 KB too.

  python3 scripts/check_apk.py app-release.apk
"""

import struct
import sys
import zipfile


def load_alignments(data: bytes) -> list[int]:
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return []
    e_phoff = struct.unpack_from("<Q", data, 0x20)[0]
    e_phentsize, e_phnum = struct.unpack_from("<HH", data, 0x36)
    out = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from("<I", data, off)[0]
        if p_type == 1:  # PT_LOAD
            out.append(struct.unpack_from("<Q", data, off + 0x30)[0])
    return out


def main() -> int:
    apk = sys.argv[1]
    bad = 0
    with zipfile.ZipFile(apk) as z, open(apk, "rb") as raw:
        total = 0
        print(f"{'library':52s} {'size':>10s} {'stored':>7s} {'align':>7s} {'offset%16K':>10s}")
        for info in z.infolist():
            if not info.filename.startswith("lib/") or not info.filename.endswith(".so"):
                continue
            data = z.read(info)
            total += info.compress_size
            aligns = load_alignments(data)
            # Data offset = local header offset + 30 + name + extra.
            raw.seek(info.header_offset)
            hdr = raw.read(30)
            n, m = struct.unpack_from("<HH", hdr, 26)
            data_off = info.header_offset + 30 + n + m
            stored = info.compress_type == zipfile.ZIP_STORED
            ok = all(a >= 16384 for a in aligns) and (not stored or data_off % 16384 == 0)
            bad += not ok
            print(f"{info.filename:52s} {len(data) / 1e6:9.2f}M {str(stored):>7s} {min(aligns) if aligns else 0:>7d} "
                  f"{data_off % 16384:>10d} {'' if ok else '  <-- not 16 KB aligned'}")
    import os
    print(f"\nAPK {os.path.getsize(apk) / 1e6:.1f} MB, native libraries {total / 1e6:.1f} MB of it")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
