import logging

# Define the log file path
log_file = "/home/creditizens/mesop/logs/agent_output.log"

# Configure the logging
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(message)s')

# Write a test log message
logging.info("Test logging write")

# Read the log file to verify the message was written
with open(log_file, 'r') as file:
    logs = file.readlines()
    for log in logs:
        print(log.strip())
