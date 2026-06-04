from transformers.models.vision_text_dual_encoder import processing_vision_text_dual_encoder
import os
import json
import subprocess
import sys
import zipfile
import pandas as pd 
from datasets import load_dataset
from sentence_transformers import SentenceTransformer   
from tqdm import tqdm


KAGGLE_DATASET = "phamtheds/job-dataset-for-recommendation"
KAGGLE_DATASET_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
HUGGINGFACE_MODEL_URL = f"https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2"
HUGGINGFACE_DATASET = "https://huggingface.co/datasets/lhoestq/resumes-raw-pdf-for-ocr"

dataset = load_dataset(HUGGINGFACE_DATASET) 
import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname("vnpdfcv"), ".cache")
from docling.document_converter import DocumentConverter
def extract_pdf_with_docling(vnpdfcv_path):
    converter = DocumentConverter()
    
    result = converter.convert(vnpdfcv_path)
    
    csv = result.document.export_to_csv()
    
    return csv


if __name__ == "__main__":
    pass



def setup_kaggle():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        return True

    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if username and key:
        os.makedirs(os.path.dirname(kaggle_json), exist_ok=True)
        with open(kaggle_json, "w", encoding="utf-8") as f:
            json.dump({"username": username, "key": key}, f)
        if os.name == 'posix':
            os.chmod(kaggle_json, 0o600)
        return True
    
    username = input("Kaggle Username: ").strip()
    key = input("Kaggle API Key: ").strip()
    if not username or not key:
        print("❌ Kaggle credentials are required. Aborting.")
        return False
    os.makedirs(os.path.dirname(kaggle_json), exist_ok=True)
    with open(kaggle_json, "w") as f:
        json.dump({"username": username, "key": key}, f)
    if os.name == 'posix':
        os.chmod(kaggle_json, 0o600)
    return True

def download_and_extract():
    target_dir = "Dataset"
    os.makedirs(target_dir, exist_ok=True)
    
    with tqdm(total=5, desc="Data Pipeline") as pbar:
        pbar.set_description("Checking Kaggle CLI")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
        pbar.update(1)
        
        pbar.set_description("Downloading from Kaggle")
        subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", target_dir], check=True)
        pbar.update(1)
            
        pbar.set_description("Extracting ZIP")
        for file in os.listdir(target_dir):
            if file.endswith('.zip'):
                zip_path = os.path.join(target_dir, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                os.remove(zip_path)
        pbar.update(1)

        pbar.set_description("Preprocessing CV")
        user_file = os.path.join(target_dir, "USER_DATA_FINAL.csv")
        if os.path.exists(user_file):
            cv = pd.read_csv(user_file)
            cv = cv.drop(columns=['URL User', 'UserID'], errors='ignore')
            if 'Desired Job' in cv.columns:
                cv['Desired Job'] = cv['Desired Job'].astype(str).str.lower().str.strip()
            
            cv = cv.rename(columns={
                'User Name': 'Tên ứng viên',
                'Desired Job': 'Vị trí ứng tuyển',
                'Industry': 'Lĩnh vực',
                'Workplace Desired': 'Nơi làm việc mong muốn',
                'Desired Salary': 'Mức lương mong muốn',
                'Gender': 'Giới tính',
                'Marriage': 'Tình trạng hôn nhân',
                'Age': 'Tuổi'
            })
            cv.to_csv(os.path.join(target_dir, "cv.csv"), index=False)
        pbar.update(1)

        pbar.set_description("Preprocessing JD")
        job_file = os.path.join(target_dir, "JOB_DATA_FINAL.csv")
        if os.path.exists(job_file):
            jd = pd.read_csv(job_file)
            jd = jd.drop(columns=['URL Job', 'JobID'], errors='ignore')
            if 'Job Title' in jd.columns:
                jd['Job Title'] = jd['Job Title'].astype(str).str.lower().str.strip()
                
            jd = jd.rename(columns={
                'Job Title': 'Vị trí cần tuyển',
                'Name Company': 'Tên công ty',
                'Company Overview': 'Giới thiệu công ty',
                'Company Size': 'Quy mô công ty',
                'Company Address': 'Địa chỉ công ty',
                'Job Description': 'Mô tả công việc',
                'Job Requirements': 'Yêu cầu công việc',
                'Benefits': 'Quyền lợi'
            })
            jd.to_csv(os.path.join(target_dir, "jd.csv"), index=False)
        pbar.update(1)
        
        pbar.set_description("Completed!")

    # Display the head(20) of each after preprocessing
    cv_out = os.path.join(target_dir, "cv.csv")
    if os.path.exists(cv_out):
        print(pd.read_csv(cv_out).head(20))
        
    jd_out = os.path.join(target_dir, "jd.csv")
    if os.path.exists(jd_out):
        print(pd.read_csv(jd_out).head(20))

if __name__ == "__main__":
    print(f"Kaggle dataset: {KAGGLE_DATASET_URL}")
    if setup_kaggle():
        download_and_extract()
