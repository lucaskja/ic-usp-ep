#######################
# Main Execution
#######################

# Upload dataset
def upload_dataset():
    """Upload dataset from local machine."""
    print("Please upload your dataset as a zip file.")
    uploaded = files.upload()
    
    if len(uploaded) == 0:
        print("No file uploaded. Please try again.")
        return None
    
    # Get the filename of the uploaded file
    filename = list(uploaded.keys())[0]
    
    # Extract the zip file
    !mkdir -p /content/datasets
    !unzip -q -o "{filename}" -d /content/datasets
    
    # Find the dataset directory
    import glob
    dataset_dirs = glob.glob('/content/datasets/*/')
    
    if len(dataset_dirs) == 0:
        print("No directories found in the uploaded zip file.")
        return None
    
    # Use the first directory as the dataset directory
    data_dir = dataset_dirs[0]
    print(f"Dataset extracted to: {data_dir}")
    
    return data_dir

# Main execution
if __name__ == "__main__":
    # Option 1: Upload dataset
    data_dir = upload_dataset()
    
    # Option 2: Use a sample dataset (uncomment to use)
    # !wget -q https://example.com/leaf_disease_dataset.zip
    # !unzip -q leaf_disease_dataset.zip -d /content/datasets
    # data_dir = '/content/datasets/leaf_disease'
    
    if data_dir:
        # Run model comparison
        run_model_comparison(
            data_dir=data_dir,
            enhanced_augmentation=True,  # Set to False for standard augmentation
            epochs=30,                   # Adjust as needed
            batch_size=32,               # Adjust based on GPU memory
            lr=0.001,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    else:
        print("Please provide a dataset to continue.")
