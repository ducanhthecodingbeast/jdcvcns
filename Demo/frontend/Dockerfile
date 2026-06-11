FROM node:20-slim

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy application code
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Start the app and expose on local network
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
