import sys
import os
from google.cloud import storage

class CloudStore(object):
    
    def __init__(self, cred_key_json_path="/app/vslml/key.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_key_json_path
    
    def upload_file(self, src_path, dst_path):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("vslml-poc-v1")
        blob = bucket.blob(dst_path)
        blob.upload_from_filename(src_path)
    
    def download_file(self, src_path, dst_path):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("vslml-poc-v1")
        blob = bucket.blob(src_path)
        blob.download_to_filename(dst_path)
