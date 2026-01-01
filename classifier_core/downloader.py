import os
import zipfile

def download_pubmed_data(data_dir="pubmed_rct"):
    """
    Downloads and extracts the PubMed 200k RCT dataset.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    def download_file(url, save_path):
        if not os.path.exists(save_path):
            print(f"Downloading {save_path}...")
            # Using curl for robustness
            ret = os.system(f'curl --http1.1 -L -o "{save_path}" "{url}"')
            if ret != 0:
                raise Exception(f"Failed to download {url}")
        else:
            print(f"{save_path} already exists.")

    files_to_download = [
        ("train.zip", "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_200k_RCT_numbers_replaced_with_at_sign/train.zip"),
        ("dev.txt", "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_200k_RCT_numbers_replaced_with_at_sign/dev.txt"),
        ("test.txt", "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_200k_RCT_numbers_replaced_with_at_sign/test.txt")
    ]

    for filename, url in files_to_download:
        save_path = os.path.join(data_dir, filename)
        download_file(url, save_path)
        
        if filename.endswith(".zip"):
            extracted_path = os.path.join(data_dir, "train.txt")
            if not os.path.exists(extracted_path):
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(save_path, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
            else:
                print(f"Extracted file {extracted_path} already exists.")
