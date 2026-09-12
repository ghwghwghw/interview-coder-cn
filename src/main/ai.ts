import { randomUUID } from 'node:crypto'
import { ipcMain } from 'electron'
import { streamText, type ModelMessage } from 'ai'
import { createOpenAI } from '@ai-sdk/openai'
import { settings, AppSettings } from './settings'

const RESPONSES_GATEWAY_HOSTS = new Set(['ai-sub2api.pudu.work', 'sub.geiliapi.com'])
const LEGACY_GATEWAY_HOST = 'ai-sub2api.pudu.work'

function shouldUseResponsesEndpoint(baseURL: string): boolean {
  try {
    return RESPONSES_GATEWAY_HOSTS.has(new URL(baseURL).hostname)
  } catch {
    return false
  }
}

function getHostname(baseURL: string): string | null {
  try {
    return new URL(baseURL).hostname
  } catch {
    return null
  }
}

function getProviderHeaders(baseURL: string): Record<string, string> | undefined {
  if (getHostname(baseURL) !== LEGACY_GATEWAY_HOST) return undefined

  return {
    'User-Agent': 'codex_cli_rs/0.147.0 (Windows 10.0.0; x86_64) powershell',
    originator: 'codex_cli_rs',
    'x-codex-session-id': randomUUID()
  }
}

function createOpenAIClient() {
  return createOpenAI({
    baseURL: settings.apiBaseURL,
    apiKey: settings.apiKey,
    headers: getProviderHeaders(settings.apiBaseURL)
  })
}

ipcMain.handle('getAvailableModels', async (_event, apiBaseURL: string, apiKey: string) => {
  if (typeof apiBaseURL !== 'string' || typeof apiKey !== 'string') {
    throw new Error('API 配置格式无效')
  }
  const baseURL = apiBaseURL.trim().replace(/\/+$/, '')
  const key = apiKey.trim()
  if (!baseURL || !key) throw new Error('请先填写 API Base URL 和 API Key')

  const response = await fetch(`${baseURL}/models`, {
    signal: AbortSignal.timeout(15_000),
    headers: {
      Authorization: `Bearer ${key}`,
      ...getProviderHeaders(baseURL)
    }
  })
  if (!response.ok) throw new Error(`获取模型失败（HTTP ${response.status}）`)

  const payload: unknown = await response.json()
  if (!payload || typeof payload !== 'object' || !('data' in payload)) {
    throw new Error('模型接口响应格式无效')
  }
  const data = (payload as { data?: unknown }).data
  if (!Array.isArray(data)) throw new Error('模型接口响应格式无效')

  const models = data.flatMap((item) => {
    if (!item || typeof item !== 'object' || !('id' in item)) return []
    const id = (item as { id?: unknown }).id
    return typeof id === 'string' && id.length > 0 ? [id] : []
  })

  if (!models.length) throw new Error('接口未返回可用模型')
  return [...new Set(models)].sort((a, b) => a.localeCompare(b))
})

// The system prompt is fully managed by the renderer (prompt scenes in the
// settings store) and synced here via updateAppSettings on app startup
function getSystemPrompt(extra?: string) {
  return [settings.customPrompt, extra].filter(Boolean).join('\n\n') || undefined
}

function getModel(_settings: AppSettings) {
  const fallbackModel = settings.apiBaseURL.includes('siliconflow')
    ? 'Qwen/Qwen3-VL-32B-Instruct'
    : 'gpt-5-mini'
  return _settings.model || fallbackModel
}

async function* getTextStream(
  messages: ModelMessage[],
  system: string | undefined,
  abortSignal?: AbortSignal
) {
  const openai = createOpenAIClient()
  const model = getModel(settings)

  const { fullStream } = streamText({
    model: shouldUseResponsesEndpoint(settings.apiBaseURL)
      ? openai.responses(model)
      : openai.chat(model),
    system,
    messages,
    abortSignal
  })

  let hasText = false
  for await (const part of fullStream) {
    if (part.type === 'error') throw part.error
    if (part.type !== 'text-delta') continue
    hasText = true
    yield part.text
  }

  if (!hasText && !abortSignal?.aborted) throw new Error('模型未返回文本内容')
}

export function getSolutionStream(messages: ModelMessage[], abortSignal?: AbortSignal) {
  return getTextStream(messages, getSystemPrompt(), abortSignal)
}

export function getFollowUpStream(
  messages: ModelMessage[],
  userQuestion: string,
  abortSignal?: AbortSignal
) {
  // Add the user's follow-up question to the conversation
  const updatedMessages: ModelMessage[] = [
    ...messages,
    {
      role: 'user',
      content: [
        {
          type: 'text',
          text: userQuestion
        }
      ]
    }
  ]

  return getTextStream(updatedMessages, getSystemPrompt(), abortSignal)
}

export function getGeneralStream(messages: ModelMessage[], abortSignal?: AbortSignal) {
  return getTextStream(
    messages,
    getSystemPrompt('注意：如果有多张截图，请结合所有截图内容进行完整分析，不要遗漏任何部分。'),
    abortSignal
  )
}
