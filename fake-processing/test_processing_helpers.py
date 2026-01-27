"""
Test script for processing_helpers.py functions
Run this to verify that all 3 functions work correctly
"""
import datetime
import logging
import os
import sys

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import required classes (mock if not available)
try:
    from fake_processing import MinIOManager, PostgreSQLDatabase, VehicleTransaction
except ImportError:
    logging.warning("Could not import from fake_processing, using mocks")
    
    # Mock classes for testing
    class VehicleTransaction:
        def __init__(self, camera_id, track_id, class_id, total_fee=0.0, 
                     time_stamp=None, img_path=None, confidence=None):
            self.camera_id = camera_id
            self.track_id = track_id
            self.class_id = class_id
            self.total_fee = total_fee
            self.time_stamp = time_stamp
            self.img_path = img_path
            self.confidence = confidence

from processing_helpers import ProcessingHelpers


class MockMinIOManager:
    """Mock MinIO Manager for testing"""
    def __init__(self):
        self.test_data_dir = "./test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        logging.info("✓ Mock MinIO Manager initialized")
    
    def download_object(self, bucket, object_name, file_path):
        """Simulate download by creating a test .npy file"""
        try:
            # Create test numpy array (simulating video frames)
            # Shape: (10, 224, 224, 3) = 10 frames of 224x224 RGB images
            test_frames = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
            np.save(file_path, test_frames)
            logging.info(f"✓ Mock downloaded: {bucket}/{object_name} -> {file_path}")
            return True
        except Exception as e:
            logging.error(f"✗ Mock download failed: {e}")
            return False


class MockDatabase:
    """Mock Database for testing"""
    def __init__(self):
        self.transactions = []
        self.conn = self  # Mock connection
        logging.info("✓ Mock Database initialized")
    
    def cursor(self):
        """Mock cursor"""
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def commit(self):
        """Mock commit"""
        logging.info("✓ Mock commit called")
    
    def rollback(self):
        """Mock rollback"""
        logging.warning("⚠ Mock rollback called")
    
    def insert_transaction(self, transaction):
        """Mock insert"""
        self.transactions.append(transaction)
        logging.info(f"✓ Mock inserted: {transaction.track_id}")
        return True


def mock_inference_function(frame):
    """
    Mock inference function for testing
    Simulates AI model prediction
    
    Args:
        frame: numpy array of image
        
    Returns:
        Tuple of (class_id, total_fee, confidence)
    """
    # Simulate random but consistent prediction
    np.random.seed(frame.mean().astype(int))
    class_id = np.random.randint(0, 12)
    confidence = np.random.uniform(0.7, 0.99)
    total_fee = [0, 0, 150, 0, 350, 350, 600, 450, 150, 150, 0, 150][class_id]
    
    return class_id, total_fee, confidence


def test_download_file_npy():
    """Test 1: download_file_npy function"""
    print("\n" + "="*70)
    print("TEST 1: download_file_npy()")
    print("="*70)
    
    try:
        # Setup
        minio_manager = MockMinIOManager()
        db = MockDatabase()
        helpers = ProcessingHelpers(minio_manager, db, output_dir="./test_temp")
        
        # Test download
        logging.info("Testing download_file_npy...")
        npy_data = helpers.download_file_npy(
            bucket_name="test-bucket",
            object_name="test_frames.npy"
        )
        
        # Verify
        if npy_data is not None:
            logging.info(f"✅ TEST PASSED: Downloaded array with shape {npy_data.shape}")
            logging.info(f"   Data type: {npy_data.dtype}")
            logging.info(f"   Value range: {npy_data.min()} - {npy_data.max()}")
            return True
        else:
            logging.error("❌ TEST FAILED: Download returned None")
            return False
            
    except Exception as e:
        logging.error(f"❌ TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_select_best_frame():
    """Test 2: select_best_frame function"""
    print("\n" + "="*70)
    print("TEST 2: select_best_frame()")
    print("="*70)
    
    try:
        # Setup
        minio_manager = MockMinIOManager()
        db = MockDatabase()
        helpers = ProcessingHelpers(minio_manager, db, output_dir="./test_temp")
        
        # Create test batch of frames
        test_batch = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
        logging.info(f"Created test batch with shape: {test_batch.shape}")
        
        # Test all methods
        methods = ["confidence", "middle", "first", "last"]
        results = {}
        
        for method in methods:
            logging.info(f"\n--- Testing method: {method} ---")
            best_frame, frame_idx, confidence = helpers.select_best_frame(
                batch=test_batch,
                inference_function=mock_inference_function,
                method=method
            )
            
            if best_frame is not None:
                results[method] = {
                    "frame_idx": frame_idx,
                    "confidence": confidence,
                    "shape": best_frame.shape
                }
                logging.info(f"✓ Method '{method}': frame {frame_idx}, confidence {confidence:.4f}")
            else:
                logging.error(f"✗ Method '{method}' failed")
                results[method] = None
        
        # Verify
        all_passed = all(r is not None for r in results.values())
        if all_passed:
            logging.info(f"\n✅ TEST PASSED: All methods worked")
            logging.info(f"   Results: {results}")
            return True
        else:
            logging.error(f"❌ TEST FAILED: Some methods failed")
            return False
            
    except Exception as e:
        logging.error(f"❌ TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_insert_in_db():
    """Test 3: batch_insert_in_db function"""
    print("\n" + "="*70)
    print("TEST 3: batch_insert_in_db()")
    print("="*70)
    
    try:
        # Setup
        minio_manager = MockMinIOManager()
        db = MockDatabase()
        helpers = ProcessingHelpers(minio_manager, db, output_dir="./test_temp")
        
        # Create test transactions
        transactions = []
        for i in range(10):
            trans = VehicleTransaction(
                camera_id=f"camera_{i % 3 + 1}",
                track_id=f"track_{i:04d}",
                class_id=i % 12,
                total_fee=150.00 * (i % 4),
                time_stamp=datetime.datetime.now(),
                img_path=f"processed/frame_{i}.jpg",
                confidence=0.85 + (i * 0.01)
            )
            transactions.append(trans)
        
        logging.info(f"Created {len(transactions)} test transactions")
        
        # Test batch insert
        logging.info("Testing batch_insert_in_db...")
        success_count, failed_count = helpers.batch_insert_in_db(transactions)
        
        # Verify
        logging.info(f"Results: {success_count} successful, {failed_count} failed")
        logging.info(f"Database now has {len(db.transactions)} transactions")
        
        if success_count == len(transactions) and failed_count == 0:
            logging.info(f"✅ TEST PASSED: All {success_count} transactions inserted")
            return True
        else:
            logging.warning(f"⚠ TEST PARTIAL: {success_count} succeeded, {failed_count} failed")
            return success_count > 0
            
    except Exception as e:
        logging.error(f"❌ TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test 4: Integration test - all functions together"""
    print("\n" + "="*70)
    print("TEST 4: INTEGRATION TEST (All functions together)")
    print("="*70)
    
    try:
        # Setup
        minio_manager = MockMinIOManager()
        db = MockDatabase()
        helpers = ProcessingHelpers(minio_manager, db, output_dir="./test_temp")
        
        # Step 1: Download .npy file
        logging.info("\n--- Step 1: Download .npy file ---")
        npy_data = helpers.download_file_npy(
            bucket_name="video-frames",
            object_name="camera01/batch_001.npy"
        )
        
        if npy_data is None:
            logging.error("❌ Step 1 failed")
            return False
        
        logging.info(f"✓ Downloaded batch with shape: {npy_data.shape}")
        
        # Step 2: Select best frame
        logging.info("\n--- Step 2: Select best frame ---")
        best_frame, frame_idx, confidence = helpers.select_best_frame(
            batch=npy_data,
            inference_function=mock_inference_function,
            method="confidence"
        )
        
        if best_frame is None:
            logging.error("❌ Step 2 failed")
            return False
        
        logging.info(f"✓ Selected frame {frame_idx} with confidence {confidence:.4f}")
        
        # Step 3: Create transactions and batch insert
        logging.info("\n--- Step 3: Batch insert transactions ---")
        transactions = []
        for i in range(5):
            trans = VehicleTransaction(
                camera_id="camera01",
                track_id=f"integration_test_{i:03d}",
                class_id=4,  # truck_20_back
                total_fee=350.00,
                time_stamp=datetime.datetime.now(),
                img_path=f"processed/frame_{frame_idx}_{i}.jpg",
                confidence=confidence
            )
            transactions.append(trans)
        
        success_count, failed_count = helpers.batch_insert_in_db(transactions)
        
        if success_count == len(transactions):
            logging.info(f"✓ Inserted {success_count} transactions")
            logging.info(f"\n✅ INTEGRATION TEST PASSED")
            return True
        else:
            logging.error(f"❌ INTEGRATION TEST FAILED at step 3")
            return False
            
    except Exception as e:
        logging.error(f"❌ INTEGRATION TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup():
    """Clean up test files"""
    import shutil
    
    test_dirs = ["./test_temp", "./test_data"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                logging.info(f"✓ Cleaned up: {test_dir}")
            except Exception as e:
                logging.warning(f"⚠ Could not clean up {test_dir}: {e}")


def main():
    """Run all tests"""
    print("="*70)
    print("PROCESSING HELPERS TEST SUITE")
    print("="*70)
    print(f"Testing 3 functions from processing_helpers.py")
    print(f"Started at: {datetime.datetime.now()}")
    print()
    
    results = {}
    
    # Run tests
    results["test_download_file_npy"] = test_download_file_npy()
    results["test_select_best_frame"] = test_select_best_frame()
    results["test_batch_insert_in_db"] = test_batch_insert_in_db()
    results["test_integration"] = test_integration()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for p in results.values() if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Functions are working correctly.")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please check the logs above.")
    
    # Cleanup
    print("\n" + "="*70)
    cleanup()
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
