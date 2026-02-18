FROM node:20-alpine AS build

WORKDIR /app/web

COPY web/package*.json ./
RUN npm install

COPY web/ ./
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM nginx:1.27-alpine

RUN apk add --no-cache curl

COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/web/dist /usr/share/nginx/html

HEALTHCHECK --interval=10s --timeout=5s --retries=5 CMD curl -f http://localhost/health || exit 1
