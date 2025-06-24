"""
Example script to run the RequiredAI server.
"""

import os
import sys
from pathlib import Path
from RequiredAI.server import RequiredAIServer

def main():
    # Path to the configuration file
    config_path = os.path.join(os.path.dirname(__file__), "server_config.json")
    
    # Create and run the server
    server = RequiredAIServer(config_path)
    server.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()
