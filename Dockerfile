# Use official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Create Streamlit config directory with proper permissions
RUN mkdir -p /tmp/.streamlit && chmod 777 /tmp/.streamlit

# Expose Streamlit's default port
EXPOSE 7860

# Set environment variables
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit
ENV STREAMLIT_WATCH_FILE=false
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]