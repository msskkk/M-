import { Env, LineWebhookBody, LineEvent, UserState, LocationData } from './types';
import { verifySignature, replyMessage, pushMessage } from './line';
import { generateLocationMessage, generateReply } from './claude';

const DEFAULT_STATE: UserState = {
  location: null,
  lastNotified: 0,
  notifyEnabled: true,
  notifyIntervalMinutes: 15,
};

async function getUserState(userId: string, kv: KVNamespace): Promise<UserState> {
  const raw = await kv.get(`user:${userId}`);
  return raw ? JSON.parse(raw) : { ...DEFAULT_STATE };
}

async function saveUserState(userId: string, state: UserState, kv: KVNamespace): Promise<void> {
  await kv.put(`user:${userId}`, JSON.stringify(state));
}

// GPS位置情報を受け取るエンドポイント（iOSショートカット等から呼ぶ）
async function handleLocationUpdate(request: Request, env: Env): Promise<Response> {
  const body = await request.json<{
    userId: string;
    latitude: number;
    longitude: number;
    address?: string;
    token?: string;
  }>();

  // 簡易認証（LINE user IDをトークン代わりに使うか、別途トークンを設定）
  if (!body.userId || !body.latitude || !body.longitude) {
    return new Response('Bad Request', { status: 400 });
  }

  const state = await getUserState(body.userId, env.KV);
  state.location = {
    latitude: body.latitude,
    longitude: body.longitude,
    address: body.address,
    timestamp: Date.now(),
  };
  await saveUserState(body.userId, state, env.KV);

  return new Response(JSON.stringify({ ok: true }), {
    headers: { 'Content-Type': 'application/json' },
  });
}

// LINE Webhookの処理
async function handleWebhook(request: Request, env: Env): Promise<Response> {
  const body = await request.text();
  const signature = request.headers.get('x-line-signature') || '';

  if (!await verifySignature(body, signature, env.LINE_CHANNEL_SECRET)) {
    return new Response('Unauthorized', { status: 401 });
  }

  const webhook: LineWebhookBody = JSON.parse(body);

  for (const event of webhook.events) {
    await handleEvent(event, env);
  }

  return new Response('OK');
}

async function handleEvent(event: LineEvent, env: Env): Promise<void> {
  if (event.type !== 'message' || !event.message) return;

  const userId = event.source.userId;
  const state = await getUserState(userId, env.KV);

  // 位置情報メッセージの場合
  if (event.message.type === 'location' && event.message.latitude && event.message.longitude) {
    state.location = {
      latitude: event.message.latitude,
      longitude: event.message.longitude,
      address: event.message.address,
      timestamp: Date.now(),
    };
    await saveUserState(userId, state, env.KV);

    const message = await generateLocationMessage(state.location, env.ANTHROPIC_API_KEY);
    await replyMessage(event.replyToken, message, env);
    return;
  }

  // テキストメッセージの場合
  if (event.message.type === 'text' && event.message.text) {
    const text = event.message.text;

    // コマンド処理
    if (text === '通知オン') {
      state.notifyEnabled = true;
      await saveUserState(userId, state, env.KV);
      await replyMessage(event.replyToken, '定期通知をオンにしたよ！', env);
      return;
    }
    if (text === '通知オフ') {
      state.notifyEnabled = false;
      await saveUserState(userId, state, env.KV);
      await replyMessage(event.replyToken, '定期通知をオフにしたよ。また必要になったら「通知オン」って送ってね', env);
      return;
    }
    if (text === '現在地') {
      if (state.location) {
        const message = await generateLocationMessage(state.location, env.ANTHROPIC_API_KEY);
        await replyMessage(event.replyToken, message, env);
      } else {
        await replyMessage(event.replyToken, '位置情報がまだ登録されてないよ。LINEから位置情報を送るか、ショートカットでGPS送信してね📍', env);
      }
      return;
    }

    // 通常の会話
    const reply = await generateReply(text, state.location, env.ANTHROPIC_API_KEY);
    await replyMessage(event.replyToken, reply, env);
  }
}

// Cron Trigger: 定期通知
async function handleScheduled(env: Env): Promise<void> {
  // KVに登録されたユーザーリストから通知対象を取得
  const userList = await env.KV.list({ prefix: 'user:' });

  for (const key of userList.keys) {
    const raw = await env.KV.get(key.name);
    if (!raw) continue;

    const state: UserState = JSON.parse(raw);
    if (!state.notifyEnabled || !state.location) continue;

    // 位置情報が古すぎる場合（6時間以上前）はスキップ
    if (Date.now() - state.location.timestamp > 6 * 60 * 60 * 1000) continue;

    // 通知間隔チェック
    const intervalMs = state.notifyIntervalMinutes * 60 * 1000;
    if (Date.now() - state.lastNotified < intervalMs) continue;

    // JST で深夜帯（23時〜7時）はスキップ
    const jstHour = new Date(Date.now() + 9 * 60 * 60 * 1000).getUTCHours();
    if (jstHour >= 23 || jstHour < 7) continue;

    const userId = key.name.replace('user:', '');
    const message = await generateLocationMessage(state.location, env.ANTHROPIC_API_KEY);
    await pushMessage(userId, message, env);

    state.lastNotified = Date.now();
    await env.KV.put(key.name, JSON.stringify(state));
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === 'POST' && url.pathname === '/webhook') {
      return handleWebhook(request, env);
    }
    if (request.method === 'POST' && url.pathname === '/location') {
      return handleLocationUpdate(request, env);
    }
    if (url.pathname === '/health') {
      return new Response('OK');
    }

    return new Response('Not Found', { status: 404 });
  },

  async scheduled(_event: ScheduledEvent, env: Env, _ctx: ExecutionContext): Promise<void> {
    await handleScheduled(env);
  },
};
