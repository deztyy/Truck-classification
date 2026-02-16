import redis
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from minio_helper import MinIOManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

MAX_DELETE_WORKERS = int(os.getenv("MAX_DELETE_WORKERS", 8))

def main():
    logging.info("🗑️  Starting Deletion Service...")

    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True
    )

    minio = MinIOManager(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
    )

    logging.info(f"✅ Connected — running {MAX_DELETE_WORKERS} concurrent delete workers")

    def delete_task(task_data: str):
        try:
            bucket, object_key = task_data.split('|', 1)
            minio.delete_object(bucket, object_key)
            logging.info(f"✅ Deleted: {bucket}/{object_key}")
        except Exception as e:
            logging.error(f"❌ Deletion failed for {task_data}: {e}")
            # Re-queue on failure
            redis_client.rpush('deletion_queue', task_data)

    with ThreadPoolExecutor(max_workers=MAX_DELETE_WORKERS) as pool:
        logging.info("👂 Listening for deletion tasks...")
        while True:
            try:
                result = redis_client.blpop('deletion_queue', timeout=5)
                if result:
                    _, task_data = result
                    pool.submit(delete_task, task_data)  # ← non-blocking, fire and forget

            except KeyboardInterrupt:
                logging.info("\n🛑 Shutting down deletion service...")
                break
            except Exception as e:
                logging.error(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()