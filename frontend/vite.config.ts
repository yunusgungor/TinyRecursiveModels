import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://backend:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Code splitting optimization
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // Vendor chunks
          if (id.includes('node_modules')) {
            if (id.includes('react') || id.includes('react-dom') || id.includes('react-router')) {
              return 'react-vendor';
            }
            if (id.includes('@tanstack/react-query')) {
              return 'query-vendor';
            }
            if (id.includes('recharts')) {
              return 'chart-vendor';
            }
            if (id.includes('@radix-ui')) {
              return 'ui-vendor';
            }
            if (id.includes('jspdf')) {
              return 'pdf-vendor';
            }
            return 'vendor';
          }
          
          // Reasoning feature chunks (lazy loaded)
          if (id.includes('/components/ReasoningPanel')) {
            return 'reasoning-panel';
          }
          if (id.includes('/components/AttentionWeightsChart') || 
              id.includes('/components/CategoryMatchingChart')) {
            return 'reasoning-charts';
          }
          if (id.includes('/components/ToolSelectionCard') || 
              id.includes('/components/ThinkingStepsTimeline')) {
            return 'reasoning-components';
          }
          
          // Core components
          if (id.includes('/components/') && !id.includes('reasoning')) {
            return 'components';
          }
        },
      },
    },
    // Optimize chunk size
    chunkSizeWarningLimit: 1000,
    // Enable minification
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true, // Remove console.log in production
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.info'], // Remove specific console methods
      },
      mangle: {
        safari10: true, // Fix Safari 10 issues
      },
    },
    // Source maps for production debugging (optional)
    sourcemap: false,
    // Optimize CSS
    cssCodeSplit: true,
    // Optimize assets
    assetsInlineLimit: 4096, // Inline assets smaller than 4kb
  },
  // Performance optimizations
  optimizeDeps: {
    include: ['react', 'react-dom', '@tanstack/react-query'],
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.stories.tsx',
        '**/*.test.tsx',
        '**/*.spec.tsx',
      ],
    },
    // Property-based testing configuration
    testTimeout: 30000, // Increased timeout for property tests (100+ iterations)
    hookTimeout: 30000,
    // Separate property tests from unit tests
    include: [
      'src/**/*.{test,spec}.{ts,tsx}',
      'src/**/*.property.{test,spec}.{ts,tsx}',
    ],
    // Exclude E2E tests (run with Playwright instead)
    exclude: [
      'node_modules/',
      'dist/',
      '.idea/',
      '.git/',
      '.cache/',
      'src/test/e2e/**',
    ],
  },
})
