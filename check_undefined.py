"""静态扫描：找出「处于 Load 上下文、但既没 import 也没定义」的名字。

这是本次 P0 事故（`SCENE_BY_HOTKEY` 未导入 → 全部热键哑掉）的通用拦截器。
compileall 只查语法，查不出未定义名字；pyflakes 能查但默认不在环境里，
所以这里用标准库 ast 做一个够用的版本，作为自测的一部分。

判定方式（保守，宁可漏报也不误报）：
  * 收集模块级 import 名、def/class 名、赋值名、参数、推导式变量、lambda 参数、
    with/except 的 as、for 目标、global/nonlocal 声明、内置名。
  * 若某 Name 处于 Load 上下文且不在上述集合里 → 报出来。
"""
from __future__ import annotations

import ast
import builtins
import io
import sys
from pathlib import Path


def defined_names(tree: ast.AST) -> set[str]:
    names: set[str] = set(dir(builtins))
    names |= set(
        "__name__ __file__ __doc__ __package__ __spec__ __loader__ __builtins__".split()
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                names.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    for sub in ast.walk(item.optional_vars):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
        elif isinstance(node, ast.comprehension):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)

    # 类型注解里的字符串前向引用 / 属性名不需要处理
    return names


def undefined_in(path: Path) -> list[tuple[int, str]]:
    try:
        src = io.open(path, encoding="utf-8-sig").read()
    except Exception:
        return []
    tree = ast.parse(src, filename=str(path))
    known = defined_names(tree)
    bad: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in known:
                bad.append((node.lineno, node.id))
    return bad


def main() -> int:
    root = Path(__file__).resolve().parent
    targets = [root / "main.py"] + sorted((root / "app").glob("*.py"))
    targets += sorted(root.glob("selftest_*.py"))
    targets += sorted((root / "tools").glob("*.py"))

    problems = 0
    for p in targets:
        bad = undefined_in(p)
        for line, name in bad:
            print(f"  [FAIL] {p.relative_to(root)}:{line} 未定义的名字 '{name}'")
            problems += 1
    if problems:
        print(f"\n结果：{problems} 处未定义名字")
        return 1
    print(f"结果：{len(targets)} 个文件，无未定义名字 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
