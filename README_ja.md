# NEO PDF AI-Optimizer  
**ドメイン制約モンテカルロ木探索（MCTS）による AI ガイド付き PDF 構造圧縮**

- 著者：Hideyoshi Murakami（村上秀吉）  
- X（旧Twitter）：@nagisa7654321  

## 概要
**NEO PDF AI-Optimizer** は、PDF 圧縮を「プリセット選択」ではなく **グローバル最適化問題** として扱う実験的な PDF 圧縮システムです。

圧縮は「一発の変換」ではなく、**ドメイン特化型 MCTS（Monte Carlo Tree Search）** により、
安全制約（壊さない・色落ちしすぎない・拡大耐性など）を守りながら、複数の圧縮戦略を探索します。

従来ツールと異なり、**各ロールアウトは実際の再圧縮パイプラインを実行**し、
その出力を **多目的・非対称スコア関数**で評価して次の探索に反映します。

## 重要な概念
- 構造化文書言語としての PDF（オブジェクト／ストリーム／XObject など）
- ドメイン制約付きモンテカルロ木探索（MCTS）
- ヒューリスティック誘導探索（画像/文字/カラー判定・安全ガード）
- 多目的・非対称評価関数（サイズ削減 × 画質/色/拡大耐性）
- Ghostscript フリーの **GS エミュレーション**（pikepdf + Pillow のみ）

## クイックスタート（ローカル）
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## デプロイ（Render）
このリポジトリには `render.yaml` が含まれます。

1. このリポジトリを GitHub にプッシュ  
2. Render で **New > Blueprint** を選択  
3. リポジトリを選択して Deploy  

---

## 実行プロファイル（FREE / STARTER）

Render の **Freeプラン**は CPU/RAM が厳しく、PDFが大きい場合や MCTS の探索が重い場合に
**502（タイムアウト）**になりやすいです。

このリポジトリでは、UIは変えずに内部上限だけを切り替えられるよう、
2つのプロファイルを用意しました（環境変数で切替）。

### 使い方（環境変数）

- Free向け（既定）:
  - `PDF_COMP_PROFILE=FREE`
- Starter向け:
  - `PDF_COMP_PROFILE=STARTER`

Render では、サービスの **Environment** から `PDF_COMP_PROFILE` を追加・変更できます。
Starter に上げたら `STARTER` に切り替えるのが目安です。

### 何が変わる？（例）

- `NEO_MAX_ITERATIONS`（探索ステップの上限）
- `NEO_MAX_THREADS`（並列スレッド上限）
- `NEO_MAX_RUNTIME_SECONDS`（全体タイムアウト）
- `NEO_MAX_UPLOAD_MB` / `NEO_MAX_PDF_PAGES`（DoS対策上限）

※ スライダーの表示（UI）はそのままですが、内部で自動的にクランプされます。
Free環境では「重い設定を選んでも中で上限がかかる」ので、落ちにくくなります。

## 免責事項
このソフトウェアは **現状有姿（AS IS）** で提供されます。  
重要書類は必ずバックアップを取り、出力結果を確認してから利用してください。

## ライセンス
Apache-2.0（`LICENSE` を参照）

## ドキュメント
- 技術フローチャート：`docs/flowchart_ja.md`（英語版: `docs/flowchart.md`）
- 日本語 README：`README_ja.md`
- 技術ノート：`TECHNICAL_NOTES.txt` / `TECHNICAL_NOTES_EXTENDED.txt`


### リポジトリ運用の注意
- **日本語などの非ASCIIファイル名は避けてください**（特に `assets/` 配下）。
  既に含まれている場合は `python tools/rename_non_ascii.py` で一括リネームできます。
  リネーム後は README の画像パス等も合わせて修正してください。
