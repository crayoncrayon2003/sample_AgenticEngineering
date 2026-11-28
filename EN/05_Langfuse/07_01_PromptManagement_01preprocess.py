import os

from dotenv import load_dotenv
from langfuse import get_client

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

#####################################
# Langfuse: Destination for execution records (traces)
#####################################
# Omitting the logging destination sends data to Langfuse Cloud. Send it to local Langfuse instead.
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
langfuse = get_client()

#####################################
# Register prompts in Langfuse
#####################################
# 01_LangChain/01_01_Chain.py embeds prompts in Python code, requiring a code change whenever a prompt changes.
# Langfuse prompt management stores prompts on the server with versions and labels.
# Version: Increments with each registration (1, 2, 3, ...); previous versions remain available.
# Label: A tag attached to a version, such as production for live use or staging for testing.
# Move a label to another version in the UI to change the prompt used by the app without changing code.
# Langfuse prompts write variables with double braces, such as {{review}}.
PROMPT_NAME = "review_extract"

PROMPT_V1 = [
    {"role": "system", "content": 'You organize product reviews.'},
    {"role": "user", "content": (
        'Extract the positives and concerns from the following product review as bullet points.\n'
        'Do not add anything absent from the review.\n\n{{review}}'
    )},
]
PROMPT_V2 = [
    {"role": "system", "content": 'You organize product reviews.'},
    {"role": "user", "content": (
        'Extract the positives and concerns from the following product review as bullet points.\n'
        'For each concern, include one suggestion for improving the product.\n'
        'Do not add anything absent from the review.\n\n{{review}}'
    )},
]


def register_prompts() -> None:
    """If unregistered, register version 1 (production) and version 2 (staging)."""
    try:
        # Do nothing if already registered, avoiding new versions on every run.
        langfuse.get_prompt(PROMPT_NAME, label="production", type="chat", cache_ttl_seconds=0)
        print(f"Prompt '{PROMPT_NAME}' is already registered.")
        return
    except Exception:
        pass
    langfuse.create_prompt(name=PROMPT_NAME, type="chat", prompt=PROMPT_V1, labels=["production"],
                           commit_message='Initial version: Extract positives and concerns')
    langfuse.create_prompt(name=PROMPT_NAME, type="chat", prompt=PROMPT_V2, labels=["staging"],
                           commit_message='Add improvement suggestions')
    print(f"Registered prompt '{PROMPT_NAME}' version 1 (production) and version 2 (staging).")


def main():
    """Example: Register product-review extraction prompts in Langfuse as preparation. 07_01_PromptManagement_02process.py retrieves and uses them. Flow: The local 07_01_PromptManagement_01preprocess.py registers two versions on the server, labeling version 1 production and version 2 staging. If already registered, do nothing; repeated runs do not add versions."""

    register_prompts()
    print(f"Prompt UI: {os.environ['LANGFUSE_BASE_URL']}/project/sample-project/prompts/{PROMPT_NAME}")


if __name__ == "__main__":
    main()
