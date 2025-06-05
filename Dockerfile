# Use official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Make startup script executable
RUN chmod +x startup.sh

# Create Streamlit config directory with proper permissions
RUN mkdir -p /tmp/.streamlit && chmod 777 /tmp/.streamlit

# Expose Streamlit's default port
EXPOSE 7860

# Set environment variables (as backup)
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit
ENV STREAMLIT_WATCH_FILE=false
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the startup script instead of streamlit directly
CMD ["./startup.sh"]