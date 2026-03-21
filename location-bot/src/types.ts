export interface Env {
  KV: KVNamespace;
  LINE_CHANNEL_ACCESS_TOKEN: string;
  LINE_CHANNEL_SECRET: string;
  ANTHROPIC_API_KEY: string;
}

export interface LocationData {
  latitude: number;
  longitude: number;
  address?: string;
  timestamp: number;
}

export interface UserState {
  location: LocationData | null;
  lastNotified: number;
  notifyEnabled: boolean;
  notifyIntervalMinutes: number;
}

// LINE Webhook Events
export interface LineWebhookBody {
  events: LineEvent[];
}

export interface LineEvent {
  type: string;
  replyToken: string;
  source: { userId: string; type: string };
  message?: LineMessage;
  timestamp: number;
}

export interface LineMessage {
  type: string;
  id: string;
  text?: string;
  latitude?: number;
  longitude?: number;
  address?: string;
}
