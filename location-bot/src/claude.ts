import Anthropic from '@anthropic-ai/sdk';
import { LocationData } from './types';

const SYSTEM_PROMPT = `あなたはユーザーの位置情報と現在時刻に基づいて、役立つ情報をカジュアルに伝えるパーソナルアシスタントです。

以下のような情報を提供してください：
- そのエリアの混雑状況（時間帯に基づく一般的な傾向）
- 飲食店のおすすめタイミング
- 電車の混雑予想・帰宅のベストタイミング
- 天気に関する一般的なアドバイス
- その他、時間帯×場所に応じた実用的な情報

トーンは友達に話しかけるようなカジュアルな日本語で。
絵文字は適度に使ってOK。
短めに（3-4文程度）まとめてください。
最初に「📍 現在地：○○周辺」「🕐 HH:MM JST」のヘッダーをつけてください。`;

export async function generateLocationMessage(
  location: LocationData,
  apiKey: string
): Promise<string> {
  const client = new Anthropic({ apiKey });
  const now = new Date();
  const jstTime = new Date(now.getTime() + 9 * 60 * 60 * 1000);
  const timeStr = jstTime.toISOString().slice(11, 16);
  const dayOfWeek = ['日', '月', '火', '水', '木', '金', '土'][jstTime.getUTCDay()];

  const userMessage = `現在の情報：
- 位置: ${location.address || `緯度${location.latitude}, 経度${location.longitude}`}
- 時刻: ${timeStr} JST（${dayOfWeek}曜日）
- 日付: ${jstTime.toISOString().slice(0, 10)}

この場所と時間帯に基づいたアドバイスをください。`;

  const response = await client.messages.create({
    model: 'claude-haiku-4-5-20251001',
    max_tokens: 300,
    system: SYSTEM_PROMPT,
    messages: [{ role: 'user', content: userMessage }],
  });

  const block = response.content[0];
  return block.type === 'text' ? block.text : '情報を取得できませんでした';
}

export async function generateReply(
  userText: string,
  location: LocationData | null,
  apiKey: string
): Promise<string> {
  const client = new Anthropic({ apiKey });
  const now = new Date();
  const jstTime = new Date(now.getTime() + 9 * 60 * 60 * 1000);
  const timeStr = jstTime.toISOString().slice(11, 16);

  const locationInfo = location
    ? `ユーザーの現在地: ${location.address || `緯度${location.latitude}, 経度${location.longitude}`}`
    : 'ユーザーの現在地: 不明';

  const systemPrompt = `あなたはユーザーのパーソナルアシスタントです。
${locationInfo}
現在時刻: ${timeStr} JST

ユーザーのメッセージに対して、位置情報や時間帯を考慮しながらカジュアルに返答してください。
日本語で、友達に話すようなトーンで。短めに返してください。`;

  const response = await client.messages.create({
    model: 'claude-haiku-4-5-20251001',
    max_tokens: 300,
    system: systemPrompt,
    messages: [{ role: 'user', content: userText }],
  });

  const block = response.content[0];
  return block.type === 'text' ? block.text : '返答を生成できませんでした';
}
