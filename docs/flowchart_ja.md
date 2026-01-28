# 技術フローチャート（MCTS PDF 圧縮）

本プロジェクトは PDF 圧縮を **探索 / 最適化問題**として扱います。
MCTS の各ロールアウトは（pikepdf + Pillow による）**実際の再圧縮パイプライン**を実行し、
サイズ削減と品質指標・安全制約を組み合わせたスコアで評価されます。

## 全体像（高レベル）

```mermaid
flowchart TD
  A[PDFアップロード] --> B[セキュリティガード
- アップロードサイズ上限
- ページ数上限
- レート制限
- 同時実行数制限]
  B --> C[文書プロファイル推定
text / mixed / image
+ カラー判定]
  C --> D[ターゲット削減率(soft)を決定
Auto or ユーザー選択]
  D --> E[Seed状態を評価
複数の初期設定で実圧縮]
  E --> F[MCTS本体]

  subgraph MCTS[MCTS探索ループ]
    F1[Selection
UCBでノード選択] --> F2[Expansion
未試行アクションで展開]
    F2 --> F3[Rollout
ランダムに数手進める]
    F3 --> F4[Execute
実際に再圧縮]
    F4 --> F5[Score
- 削減率
- 画像MSEベース指標
- カラー保持係数
- 拡大耐性係数
- gate (閾値)]
    F5 --> F6[Backprop
平均スコアを親へ]
  end

  F --> G[ベスト状態を再実行]
  G --> H[ダウンロード]

  %% Compression internals
  subgraph CORE[圧縮コア（GS-Emulation / pike-only）]
    X1[メタデータ削除]
    X2[画像XObject抽出]
    X3[画像種別判定
photo / texty / mono/gray/color]
    X4[ダウンサンプル判定
（DPI・閾値）]
    X5[エンコード選択
JPEG or Flate(1bit/8bit)]
    X6[小さくなる場合のみ置換]
    X7[Contents/Formの単純Flate再圧縮]
    X8[pikepdf save
object streams / compress_streams]
  end

  F4 --> CORE
  CORE --> F5

  %% Cache
  subgraph CACHE[キャッシュ（セッション分離）]
    K1[(session_id + orig_hash + state_key)] --> K2[結果バイト/プレビュー保持]
  end
  F4 --> CACHE
```

## 重要ポイント

- **UIは変えずに**、内部の上限（探索数・スレッド・実行時間など）を環境変数でクランプします。
- **FREE / STARTER プロファイル**は `PDF_COMP_PROFILE` で切替（README参照）。
- `engine="gs"` は Ghostscript 実行ではなく、**GSプリセットを模倣した pike-only 挙動**です（GS-Emulation）。
- 画像は **PdfImage → PIL** で扱い、JPEG/Flate への再エンコードを行います。
- 置換は原則「小さくなるなら採用」。品質側はスコア関数がトレードオフを吸収します。
