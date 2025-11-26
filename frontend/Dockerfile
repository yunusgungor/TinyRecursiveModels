# syntax=docker/dockerfile:1.4
# Multi-stage build for React frontend with BuildKit optimizations

# Base stage - minimal node alpine image
FROM node:20-alpine as base

# Set working directory
WORKDIR /app

# Set environment variables for production builds
ENV NODE_ENV=production

# Dependencies stage - copy package files for cache optimization
FROM base as dependencies

# Copy only package files first (cache optimization)
# This ensures dependency layer is cached separately from code
COPY package.json package-lock.json ./

# Development stage - includes dev dependencies and hot reload
FROM dependencies as development

# Override NODE_ENV for development
ENV NODE_ENV=development

# Use cache mount for npm to speed up builds
# This persists npm cache across builds
RUN --mount=type=cache,target=/root/.npm \
    npm ci

# Copy application code
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Start development server with hot reload
CMD ["npm", "run", "dev", "--", "--host"]

# Builder stage - builds production assets
FROM dependencies as builder

# Use cache mount for npm
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production

# Copy application code
COPY . .

# Build production assets
RUN npm run build

# Production stage - nginx serving static files
FROM nginx:alpine as production

# Copy custom nginx configuration for SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Security: Configure non-root nginx user and set minimal file permissions
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx.pid && \
    # Set minimal file permissions (least privilege principle)
    # Static files should be readable but not writable or executable
    find /usr/share/nginx/html -type f -exec chmod u=r,g=r,o= {} \; && \
    find /usr/share/nginx/html -type d -exec chmod u=rx,g=rx,o= {} \; && \
    # Nginx config should be readable only
    chmod u=r,g=r,o= /etc/nginx/conf.d/default.conf && \
    # Cache and log directories need write permissions for nginx user
    chmod u=rwx,g=rx,o= /var/cache/nginx && \
    chmod u=rwx,g=rx,o= /var/log/nginx

# Switch to non-root user
USER nginx

# Expose HTTP port
EXPOSE 80

# Start nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
