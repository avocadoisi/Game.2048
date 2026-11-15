#!/usr/bin/env python3
"""issue に area/kind ラベルを付与する（#282）。

- 既定は dry-run。`--apply` で GitHub を書き換える。
- 冪等: 既に付いているラベルは触らない。2 回流すと差分は出ない。
- ラベルの定義（色・説明）はこのファイルが持ち、無ければ作る。
- どの issue に何を付けるかは `scripts/issue-labels.tsv`。

usage:
  python3 scripts/label-issues.py            # 何が変わるか表示
  python3 scripts/label-issues.py --apply    # 実行
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "scripts" / "issue-labels.tsv"

# (name, color, description)
LABELS = [
    ("area:solver", "1d76db", "探索と評価関数（expectimax / 方策 / n-tuple / 枝刈り）"),
    ("area:nn", "5319e7", "NN の重み・学習則・蒸留・推論"),
    ("area:eval", "0e8a16", "計測基盤と判定手順（pairw / ranking / benchmark / evalab の土台）"),
    ("area:core", "b60205", "ゲームロジック 2048.core（move / merge / row_table）"),
    ("area:web", "fbca04", "wasm・ブラウザ UI・配信・HTTP"),
    ("area:docs", "0075ca", "docs / README / skill"),
    ("kind:explore", "c2e0c6", "仮説を試して採否を出す（未知の効果を測る）"),
    ("kind:implement", "a2eeef", "コードやノブを足す・差し替える（挙動不変の整理も含む）"),
    ("kind:measure", "d4c5f9", "既にある構成を測り直す・数値を確定する"),
    ("kind:fix", "d73a4a", "壊れているものを直す（ビルド・テスト・計測の誤り）"),
    ("kind:docs", "fef2c0", "ドキュメントを書く・直す"),
]


def gh(*args, capture=True):
    return subprocess.run(
        ["gh", *args],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=capture,
    )


def load_table():
    rows = {}
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        num, area, kind, *_note = line.split("\t")
        rows[int(num)] = (area, kind)
    return rows


def issues():
    out = gh("issue", "list", "--state", "all", "--limit", "1000",
             "--json", "number,title,labels").stdout
    return json.loads(out)


def ensure_labels(apply):
    have = {l["name"] for l in json.loads(gh("label", "list", "--limit", "200", "--json", "name").stdout)}
    for name, color, desc in LABELS:
        if name in have:
            continue
        print(f"{'' if apply else 'would create label: '}label {name} ({color})")
        if apply:
            gh("label", "create", name, "--color", color, "--description", desc)


def main():
    apply = "--apply" in sys.argv
    unknown = [a for a in sys.argv[1:] if a not in ("--apply",)]
    if unknown:
        sys.exit(f"unknown argument: {unknown[0]} (usage: label-issues.py [--apply])")

    rows = load_table()
    ensure_labels(apply)

    items = issues()
    todo, unlabeled, unmapped = 0, 0, []
    for it in items:
        num = it["number"]
        have = {l["name"] for l in it["labels"]}
        if not have:
            unlabeled += 1
        if num not in rows:
            if not have:
                unmapped.append((num, it["title"]))
            continue
        area, kind = rows[num]
        need = [n for n in (f"area:{area}", f"kind:{kind}") if n not in have]
        if not need:
            continue
        todo += 1
        print(f"{'' if apply else 'would run: '}#{num} +{','.join(need)}  {it['title'][:50]}")
        if apply:
            gh("issue", "edit", str(num), "--add-label", ",".join(need))

    print()
    print(f"ラベル: {len(LABELS)} 個 / table: {len(rows)} 件 / GitHub: {len(items)} 件")
    if apply:
        left = [it["number"] for it in issues() if not it["labels"]]
        print(f"ラベル無し: {unlabeled} 件 -> {len(left)} 件" + (f"  {left}" if left else ""))
        print(f"付与: {todo} 件")
    else:
        print(f"ラベル無し: {unlabeled} 件 / 付与予定: {todo} 件（--apply で実行）")
    for num, title in unmapped:
        print(f"warn: #{num} は table に無く、ラベルも無い。GitHub 側で直接付ける: {title[:50]}")
    return 1 if apply and (left or unmapped) else 0


if __name__ == "__main__":
    sys.exit(main())
