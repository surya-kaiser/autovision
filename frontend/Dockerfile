# Frontend Dockerfile — AutoVision React + Nginx
# Multi-stage build: build in Node, serve with Nginx (production-optimized)

# ── Builder stage: Node environment for npm build ──────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files first for Docker cache optimization
COPY package.json package-lock.json ./

# Install dependencies (cached layer)
RUN npm ci --prefer-offline --no-audit

# Copy source code
COPY . .

# Build React application
RUN npm run build

# ── Server stage: Nginx for lightweight production serving ─────────────────
FROM nginx:alpine

# Install curl for health checks
RUN apk add --no-cache curl

# Copy built application from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check (checks if nginx is responding)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
