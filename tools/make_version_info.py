"""生成 Windows 版本资源文件 build/version_info.txt。

PyInstaller 的 version 参数需要这种格式，用来在 exe 属性里显示
「产品名称 / 版本 / 版权」，而不是一片空白。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.version import APP_NAME, APP_NAME_CN, HOMEPAGE, ORG, __version__  # noqa: E402

parts = (__version__.split(".") + ["0", "0", "0", "0"])[:4]
nums = ", ".join(str(int(p)) for p in parts)

content = f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({nums}),
    prodvers=({nums}),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '080404b0',
        [StringStruct('CompanyName', '{ORG}'),
         StringStruct('FileDescription', '{APP_NAME_CN} / {APP_NAME}'),
         StringStruct('FileVersion', '{__version__}'),
         StringStruct('InternalName', 'EQBandMonitor'),
         StringStruct('LegalCopyright', 'MIT License'),
         StringStruct('OriginalFilename', 'EQBandMonitor.exe'),
         StringStruct('ProductName', '{APP_NAME}'),
         StringStruct('ProductVersion', '{__version__}'),
         StringStruct('Comments', '{HOMEPAGE}')])
    ]),
    VarFileInfo([VarStruct('Translation', [2052, 1200])])
  ]
)
"""

out = ROOT / "build" / "version_info.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(content, encoding="utf-8")
print(f"wrote {out} ({len(content)} chars) version={__version__}")
