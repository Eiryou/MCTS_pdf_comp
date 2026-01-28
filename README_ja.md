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

## 免責事項
このソフトウェアは **現状有姿（AS IS）** で提供されます。  
重要書類は必ずバックアップを取り、出力結果を確認してから利用してください。

## ライセンス
Apache-2.0（`LICENSE` を参照）

## ドキュメント
- 技術フローチャート：`docs/flowchart.md`
- 日本語 README：`README_ja.md`
- 技術ノート：`TECHNICAL_NOTES.txt` / `TECHNICAL_NOTES_EXTENDED.txt`
