import os

# Automatically bind to the PORT environment variable provided by Render
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"

# Optimize workers and timeout for Render
workers = 2
timeout = 120
