# location-bot

位置情報×時間帯ベースのパーソナル通知LINEボット。
Cloudflare Workers + Claude API で動作。

## 機能

- **定期通知**: 15分ごとに現在地周辺の情報（混雑・飲食店・電車）をプッシュ
- **位置情報受信**: LINE位置情報送信 or iOSショートカットからGPS自動送信
- **会話応答**: テキストメッセージに位置・時間を考慮して返答
- **コマンド**: `通知オン` / `通知オフ` / `現在地`

## セットアップ

### 1. LINE Developers

1. [LINE Developers Console](https://developers.line.biz/) でチャネル作成
2. Messaging API チャネルを選択
3. チャネルアクセストークン（長期）を発行
4. チャネルシークレットをメモ

### 2. Cloudflare Workers

```bash
cd location-bot
npm install

# KV Namespace 作成
wrangler kv:namespace create KV
# 出力されたIDを wrangler.toml に設定

# シークレット設定
npm run secret:line-token
npm run secret:line-secret
npm run secret:anthropic

# デプロイ
npm run deploy
```

### 3. LINE Webhook 設定

デプロイ後のURL `https://location-bot.<your-subdomain>.workers.dev/webhook` を
LINE Developers Console の Webhook URL に設定。

### 4. iOSショートカットでGPS自動送信

iOSの「ショートカット」アプリで以下のオートメーションを作成：

1. トリガー: 特定の時間 or 場所の到着/出発
2. アクション:
   - 「現在地を取得」
   - 「URLの内容を取得」で以下にPOST:

```
URL: https://location-bot.<your-subdomain>.workers.dev/location
Method: POST
Body (JSON):
{
  "userId": "<LINE_USER_ID>",
  "latitude": (ショートカット変数),
  "longitude": (ショートカット変数),
  "address": (ショートカット変数)
}
```

LINE User IDはボットにメッセージを送った際のWebhookログから確認可能。
