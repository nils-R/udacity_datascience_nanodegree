# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory to /disaster
WORKDIR /disaster

# Copy the current directory contents into the container at /disaster
COPY . /disaster

# Expose port 5000
EXPOSE 5000

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run run.py when the container launches
CMD ["python", "app/run.py"]