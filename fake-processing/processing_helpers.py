"""
Processing Helper Functions
Contains utility functions for frame processing, database operations, and file handling
"""
import datetime
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import psycopg2.extras


class ProcessingHelpers:
    """Helper class containing processing utility functions"""
    
    def __init__(self, minio_manager, db, output_dir: str = "./processed_data"):
        """
        Initialize ProcessingHelpers
        
        Args:
            minio_manager: MinIOManager instance
            db: PostgreSQLDatabase instance
            output_dir: Directory for temporary files
        """
        self.minio_manager = minio_manager
        self.db = db
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def download_file_npy(
        self,
        bucket_name: str,
        object_name: str,
        local_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Download .npy file from MinIO and load as numpy array
        
        Args:
            bucket_name: MinIO bucket name
            object_name: Object key/path in MinIO
            local_path: Optional local path to save file (if None, uses temp file)
            
        Returns:
            np.ndarray if successful, None otherwise
        """
        try:
            # Create temp file path if not provided
            if local_path is None:
                local_path = os.path.join(
                    self.output_dir, 
                    f"temp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(object_name)}"
                )
            
            # Download from MinIO
            logging.info(f"Downloading {bucket_name}/{object_name} to {local_path}")
            success = self.minio_manager.download_object(
                bucket=bucket_name,
                object_name=object_name,
                file_path=local_path
            )
            
            if not success:
                logging.error(f"Failed to download {object_name} from MinIO")
                return None
            
            # Load numpy array
            npy_data = np.load(local_path)
            logging.info(f"✓ Loaded .npy file with shape: {npy_data.shape}")
            
            # Clean up temp file
            if os.path.exists(local_path):
                os.remove(local_path)
                logging.debug(f"Cleaned up temp file: {local_path}")
            
            return npy_data
            
        except Exception as e:
            logging.error(f"✗ Error downloading .npy file: {e}")
            # Clean up on error
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
            return None
    
    def select_best_frame(
        self,
        batch: np.ndarray,
        inference_function,
        method: str = "confidence"
    ) -> Tuple[Optional[np.ndarray], int, float]:
        """
        Select the best frame from a batch using AI model inference
        
        Args:
            batch: Numpy array of frames, shape (N, H, W, C) or single frame (H, W, C)
            inference_function: Function to run inference on a frame, returns (class_id, fee, confidence)
            method: Selection method - "confidence" (highest), "middle" (center frame), "first", "last"
            
        Returns:
            Tuple of (best_frame, frame_index, confidence_score)
        """
        try:
            # Normalize frame to uint8
            def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
                """Convert array to uint8, normalizing floats when appropriate."""
                if arr.dtype in (np.float32, np.float64):
                    if arr.max() <= 1.0:
                        return (arr * 255).astype(np.uint8)
                    return arr.astype(np.uint8)
                if arr.dtype != np.uint8:
                    return arr.astype(np.uint8)
                return arr
            
            # Handle single frame
            if batch.ndim == 3:
                logging.info("Single frame provided, running inference")
                frame = normalize_to_uint8(batch)
                class_id, _, confidence = inference_function(frame)
                return frame, 0, confidence
            
            # Handle batch of frames
            if batch.ndim != 4:
                logging.error(f"Invalid batch shape: {batch.shape}. Expected (N, H, W, C)")
                return None, -1, 0.0
            
            num_frames = batch.shape[0]
            logging.info(f"Selecting best frame from {num_frames} frames using method: {method}")
            
            # Method: middle frame (fast, no inference needed)
            if method == "middle":
                frame_idx = num_frames // 2
                frame = normalize_to_uint8(batch[frame_idx])
                _, _, confidence = inference_function(frame)
                logging.info(f"Selected middle frame {frame_idx} with confidence: {confidence:.4f}")
                return frame, frame_idx, confidence
            
            # Method: first frame
            elif method == "first":
                frame = normalize_to_uint8(batch[0])
                _, _, confidence = inference_function(frame)
                logging.info(f"Selected first frame with confidence: {confidence:.4f}")
                return frame, 0, confidence
            
            # Method: last frame
            elif method == "last":
                frame_idx = num_frames - 1
                frame = normalize_to_uint8(batch[frame_idx])
                _, _, confidence = inference_function(frame)
                logging.info(f"Selected last frame with confidence: {confidence:.4f}")
                return frame, frame_idx, confidence
            
            # Method: highest confidence (default) - run inference on all frames
            else:  # method == "confidence"
                best_frame = None
                best_idx = -1
                best_confidence = 0.0
                
                for i in range(num_frames):
                    frame = normalize_to_uint8(batch[i])
                    _, _, confidence = inference_function(frame)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_frame = frame
                        best_idx = i
                
                logging.info(
                    f"✓ Selected frame {best_idx}/{num_frames-1} with highest confidence: {best_confidence:.4f}"
                )
                return best_frame, best_idx, best_confidence
                
        except Exception as e:
            logging.error(f"✗ Error selecting best frame: {e}")
            import traceback
            traceback.print_exc()
            return None, -1, 0.0
    
    def batch_insert_in_db(
        self,
        transactions: List
    ) -> Tuple[int, int]:
        """
        Insert multiple transactions into database in a single batch operation
        
        Args:
            transactions: List of VehicleTransaction objects to insert
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not transactions:
            logging.warning("No transactions to insert")
            return 0, 0
        
        successful = 0
        failed = 0
        
        try:
            with self.db.conn.cursor() as cur:
                # Prepare batch data
                batch_data = [
                    (
                        t.camera_id,
                        t.track_id,
                        t.class_id,
                        t.total_fee,
                        t.time_stamp or datetime.datetime.now(datetime.timezone.utc),
                        t.img_path,
                        t.confidence
                    )
                    for t in transactions
                ]
                
                # Execute batch insert using execute_values for better performance
                insert_query = """
                    INSERT INTO vehicle_transactions 
                    (camera_id, track_id, class_id, total_fee, time_stamp, img_path, confidence)
                    VALUES %s
                """
                
                psycopg2.extras.execute_values(
                    cur,
                    insert_query,
                    batch_data,
                    template="(%s, %s, %s, %s, %s, %s, %s)"
                )
                
                self.db.conn.commit()
                successful = len(transactions)
                
                logging.info(
                    f"✓ Batch insert successful: {successful} transactions inserted"
                )
                
        except Exception as e:
            self.db.conn.rollback()
            logging.error(f"✗ Batch insert failed: {e}")
            
            # Fallback: try inserting one by one
            logging.info("Attempting individual inserts as fallback...")
            for transaction in transactions:
                try:
                    if self.db.insert_transaction(transaction):
                        successful += 1
                    else:
                        failed += 1
                except Exception as e2:
                    logging.error(f"Failed to insert transaction {transaction.track_id}: {e2}")
                    failed += 1
        
        logging.info(
            f"Batch insert complete: {successful} successful, {failed} failed"
        )
        return successful, failed
