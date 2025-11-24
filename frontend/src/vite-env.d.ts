/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  readonly VITE_APP_NAME: string
  readonly VITE_APP_VERSION: string
  readonly VITE_API_BASE_URL: string
  readonly VITE_ENABLE_REASONING: string
  readonly VITE_DEFAULT_REASONING_LEVEL: string
  readonly VITE_ENABLE_REASONING_EXPORT: string
  readonly VITE_ENABLE_REASONING_COMPARISON: string
  readonly VITE_MAX_THINKING_STEPS: string
  readonly VITE_REASONING_CACHE_TTL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
