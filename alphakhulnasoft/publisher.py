import os

from huggingface_hub import HfApi


class HFPublisher:
    """Publishes benchmark results to Hugging Face Hub."""

    def __init__(self, repo_id: str, token: str | None = None):
        self.repo_id = repo_id
        self.api = HfApi(token=token or os.getenv("HF_TOKEN"))

    def publish_results(self, results_file: str):
        """Uploads a results JSON file to a HF dataset repo."""
        if not os.path.exists(results_file):
            print(f"❌ Error: File {results_file} not found.")
            return

        print(f"🚀 Uploading {results_file} to Hugging Face repo: {self.repo_id}...")
        try:
            filename = os.path.basename(results_file)
            self.api.upload_file(
                path_or_fileobj=results_file,
                path_in_repo=f"results/{filename}",
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Upload benchmark results: {filename}",
            )
            print(
                f"✅ Successfully published to HF: https://huggingface.co/datasets/{self.repo_id}"
            )
        except Exception as e:
            print(f"❌ Failed to publish to HF: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m alphakhulnasoft.publisher <results_file> <hf_repo_id>")
        sys.exit(1)

    publisher = HFPublisher(sys.argv[2])
    publisher.publish_results(sys.argv[1])
