import kagglehub

# Download latest version
path = kagglehub.dataset_download("atomscott/teamtrack")

print("Path to dataset files:", path)