# FallGuard Lite

人物検出と時系列の姿勢変化を組み合わせ、**転倒の可能性**をブラウザ上で可視化する小型Webアプリです。カメラまたはローカル動画を選ぶと、人物の枠・姿勢点・状態（通常／要確認／転倒の可能性／回復確認中）を表示します。

> [!WARNING]
> 技術デモであり、医療機器・介護機器・生命安全システムではありません。誤検知と見逃しがあり得ます。緊急通報、救助、見守りの唯一の判断材料には使用しないでください。

## 特長

- YOLO11n Poseで人物と17点の姿勢キーポイントを検出
- 接続ごとの軽量IoUトラッカーで複数人物の履歴を分離
- 立位履歴、重心の急降下、体幹角度、横長の人物枠、横姿勢の継続時間を統合
- `normal → suspected → fallen → recovering` の状態機械で瞬間的な誤検知を抑制
- カメラ映像・動画・JPEGフレームをサーバー側へ保存しない
- CPUのみで動作し、Docker Composeでローカルホストに限定公開
- 主要な姿勢遷移を実画像なしの決定論的テストで検証

## すぐに試す

必要なものはDocker Desktop（またはDocker Engine + Compose）です。

```bash
docker compose up --build
```

起動後、<http://localhost:8000> を開いて「カメラを開始」または「動画を開く」を選びます。初回ビルドでは依存パッケージと `yolo11n-pose.pt` を取得するため、数分かかる場合があります。停止は次のコマンドです。

```bash
docker compose down
```

Composeは `127.0.0.1:8000` のみにバインドします。LANやインターネットへ公開する設計ではありません。カメラAPIは `localhost` またはHTTPSの安全なコンテキストで利用してください。

## 判定アルゴリズム

各人物について、次の観測値を秒単位で追跡します。

1. 立位の人物枠高さの中央値を基準として蓄積（既定1.0秒）
2. 人物中心の下向き速度が基準身長の0.65倍/秒以上か確認
3. 人物枠の縦横比、肩―腰の体幹角度、高さ低下から横姿勢スコアを計算
4. 急降下後の横姿勢が0.75秒続いた場合に `fallen` を確定
5. 立位が1.0秒続けば `recovering` から復帰し、2.5秒の再通知クールダウンを適用

起動時から横たわっている人には立位履歴がないため通知しません。しゃがみ、着席、ゆっくり横になる動作は「急降下＋横姿勢継続」の組み合わせを満たしにくくしています。ただしカメラ角度、遮蔽、極端に広い服、人物の交差などで誤判定する可能性があります。

主な調整値は [`.env.example`](.env.example) にあります。実環境では実際のカメラ位置と対象動作で評価し、値を調整してください。

## プライバシーとデータフロー

```text
Camera / local video
        │  sampled JPEG (max 8 fps, max 960 px wide)
        ▼
Same-origin WebSocket ──► in-memory decode ──► pose + fall state
        ▲                                           │
        └──────────── result JSON only ─────────────┘
```

- 動画ファイル全体をアップロードせず、ブラウザで再生したフレームだけを送信します。
- バックエンドはフレーム、顔画像、動画、Base64データをファイルやログへ保存しません。
- 解析結果も永続化しません。ブラウザを停止・再読込すると状態は消えます。
- アクセスログは既定で無効です。
- 顔の匿名化・認証・録画保管機能はありません。

## 構成

```text
app/
  main.py             FastAPI / WebSocket / 入力制限
  detector.py         YOLO pose / 接続別人物追跡
  fall_detector.py    依存なしの時系列状態機械
  static/             ブラウザUI
tests/                合成座標とAPIのテスト
Dockerfile            非root CPUイメージ
docker-compose.yml    localhost限定・read-only実行
```

## 開発

Python 3.11以上を使用します。通常のローカル環境では、最初の推論時にUltralyticsの公式配布元からモデルが取得されます。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

品質チェック:

```bash
ruff check .
ruff format --check .
pytest
```

モデルを手動配置する場合は `MODEL_PATH` をそのファイルへ向け、`MODEL_SHA256` に期待する小文字SHA-256を設定できます。DockerイメージはUltralytics Assetsの `v8.4.0/yolo11n-pose.pt` を取得し、SHA-256 `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` と照合します。起動時にも隣接する `.sha256` ファイルを検証します。

## API

- `GET /` — Web UI
- `GET /health` — サービスとモデル読込状態（モデル自体は読込まない）
- `WS /ws/analyze` — JPEGバイナリを受け、人物・姿勢点・状態のJSONを返す
- `GET /api/docs` — FastAPIが生成するHTTP APIドキュメント

WebSocketは、Originヘッダーがある場合、既定でHostヘッダーと同じホスト・ポートのHTTP/HTTPSオリジンを許可します。リバースプロキシ構成では `ALLOWED_ORIGINS=https://example.com` のように追加するオリジンを明示してください。この設定はHostに基づく許可を置き換えません。Originなしのプログラム接続は許可され、この検査は認証や接続先Hostの制限ではありません。localhost限定で使用してください。

アプリが処理するJPEGは2.5 MBまでです。JPEGヘッダーで約2.1メガピクセルの上限をデコード前に検査し、デコード後も再確認します。寸法が未確定のJPEGは受け付けません。最短間隔50 msの制限は接続ごとで、全接続を合計した資源上限ではありません。

## 既知の制約

- CPU推論速度はホスト性能に依存し、表示上の8 FPSを下回る場合があります。
- 単眼RGB映像なので実世界の高さや床面を直接推定しません。
- 強い遮蔽、暗所、魚眼、真上からの撮影、画面外への転倒は苦手です。
- IoU中心の簡易トラッカーなので人物が密に交差するとIDが入れ替わる場合があります。
- 「ゆっくり倒れる」「椅子から滑り落ちる」など急降下が弱い転倒を見逃す場合があります。
- 実運用前には対象環境の動画で感度・特異度を測定し、別系統の確認手段を必ず用意してください。

## ライセンス

本リポジトリは **GNU AGPL-3.0-or-later** です。UltralyticsのパッケージとモデルにもAGPL-3.0または別途Enterprise条件が適用されます。商用・閉源・ホステッド利用の前に必ず現在のUltralyticsライセンスを確認してください。詳細は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) を参照してください。
