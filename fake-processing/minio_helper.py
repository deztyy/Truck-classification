import logging
from io import BytesIO
from typing import List, Dict, Optional
from minio import Minio
from minio.error import S3Error

class MinIOManager:
    """Manages MinIO operations"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        self.endpoint = endpoint
        self.client = Minio(
            endpoint=endpoint, 
            access_key=access_key, 
            secret_key=secret_key, 
            secure=secure
        )
        logging.info(f"✓ MinIO connected: {endpoint}")

    def delete_object(self, bucket: str, object_name: str) -> bool:
        """Delete a single object from MinIO"""
        try:
            self.client.remove_object(bucket, object_name)
            return True
        except S3Error as e:
            logging.error(f"✗ Delete failed: {e}")
            return False

    def create_bucket(self, bucket_name: str) -> bool:
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
            return True
        except S3Error as e:
            logging.error(f"✗ Create bucket failed: {e}")
            return False

    def upload_from_bytes(self, bucket: str, object_name: str, data: bytes, 
                          content_type: str = "application/octet-stream") -> bool:
        try:
            self.client.put_object(
                bucket, object_name, BytesIO(data), 
                length=len(data), content_type=content_type
            )
            return True
        except S3Error as e:
            logging.error(f"✗ Upload failed: {e}")
            return False

    def get_object_data(self, bucket: str, object_name: str) -> bytes:
        response = None
        try:
            response = self.client.get_object(bucket, object_name)
            return response.read()
        except S3Error as e:
            logging.error(f"✗ Download failed: {e}")
            raise
        finally:
            if response:
                response.close()
                response.release_conn()