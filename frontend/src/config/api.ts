/// <reference types="../vite-env.d.ts" />

/**
 * API configuration for reasoning-enhanced recommendations
 */

export interface ApiConfig {
  baseUrl: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
}

/**
 * Get API configuration from environment variables
 */
export const getApiConfig = (): ApiConfig => {
  return {
    baseUrl: import.meta.env.VITE_API_BASE_URL || '/api',
    timeout: 30000, // 30 seconds for reasoning requests
    retryAttempts: 3,
    retryDelay: 1000, // 1 second
  };
};

/**
 * API configuration instance
 */
export const apiConfig = getApiConfig();

/**
 * API endpoints
 */
export const API_ENDPOINTS = {
  recommendations: '/v1/recommendations',
  tools: '/v1/tools',
  health: '/v1/health',
} as const;

/**
 * Get full API URL
 */
export const getApiUrl = (endpoint: string): string => {
  return `${apiConfig.baseUrl}${endpoint}`;
};
