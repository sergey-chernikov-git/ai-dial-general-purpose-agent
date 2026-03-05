from os import getenv

TOOL_CALL_HISTORY_KEY = "tool_call_history"
CUSTOM_CONTENT = "custom_content"
API_KEY = getenv('API_KEY', '')
API_VERSION = getenv('API_VERSION', '2025-01-01-preview')
API_ENDPOINT = getenv('API_ENDPOINT', 'https://ai-proxy.lab.epam.com')