"""Upload the model and the Space to the Hugging Face Hub."""

from huggingface_hub import HfApi

USERNAME = "Gaelletheresa"
MODEL_REPO = f"{USERNAME}/lakeshore-ticket-sorter"
SPACE_REPO = f"{USERNAME}/lakeshore-ticket-sorter-demo"


def publish_model():
    """Create the model repository and upload the model folder with its card."""
    api = HfApi()
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path="model", repo_id=MODEL_REPO, repo_type="model")
    print(f"Model: https://huggingface.co/{MODEL_REPO}")


def publish_space():
    """Create the Space and upload the three files from the space folder."""
    api = HfApi()
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path="space", repo_id=SPACE_REPO, repo_type="space")
    print(f"Space: https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    import sys

    if "--space" in sys.argv:
        publish_space()
    else:
        publish_model()
