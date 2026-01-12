# Stage 1: Build
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package.json yarn.lock ./

# Install dependencies
RUN yarn install --frozen-lockfile

# Copy source code
COPY . .

# Build argument for backend URL
ARG REACT_APP_BACKEND_URL=http://localhost:8000
ENV REACT_APP_BACKEND_URL=$REACT_APP_BACKEND_URL

# Build the application
RUN yarn build

# Stage 2: Serve
FROM nginx:alpine

# Copy build artifacts from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration (optional, using default for now but recommended to handle React Router)
RUN echo 'server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
