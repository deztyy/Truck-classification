#!/usr/bin/env python
"""
สคริปต์ทดสอบว่า ONNX model detect รถได้หรือไม่
"""
import numpy as np
import cv2
import onnxruntime as ort
import glob
import os

# Vehicle classes
VEHICLE_CLASSES = {
    0: "car",
    1: "other",
    2: "other_truck",
    3: "pickup_truck",
    4: "truck_20_back",
    5: "truck_20_front",
    6: "truck_20x2",
    7: "truck_40",
    8: "truck_roro",
    9: "truck_tail",
    10: "motorcycle",
    11: "truck_head",
}

def preprocess_frame(frame_bgr, input_size=(640, 640)):
    """Preprocess BGR frame for ONNX model"""
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize
    frame_resized = cv2.resize(frame_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    frame_norm = frame_resized.astype(np.float32) / 255.0
    
    # Convert to CHW format
    frame_chw = np.transpose(frame_norm, (2, 0, 1))
    frame_chw = np.expand_dims(frame_chw, axis=0)
    
    return frame_chw

def postprocess_simple(outputs, conf_thresh=0.25):
    """Simple postprocessing to check detections"""
    output = outputs[0][0]  # [84, 8400]
    
    boxes = output[:4, :].T
    class_probs = output[4:, :].T
    
    class_ids = np.argmax(class_probs, axis=1)
    confidences = np.max(class_probs, axis=1)
    
    print(f"\n📊 Model Output Stats:")
    print(f"   - Total predictions: {len(confidences)}")
    print(f"   - Max confidence: {confidences.max():.4f}")
    print(f"   - Min confidence: {confidences.min():.4f}")
    print(f"   - Mean confidence: {confidences.mean():.4f}")
    
    # Filter by threshold
    mask = confidences > conf_thresh
    filtered_boxes = boxes[mask]
    filtered_class_ids = class_ids[mask]
    filtered_confidences = confidences[mask]
    
    print(f"\n✅ Detections above {conf_thresh} threshold: {len(filtered_confidences)}")
    
    if len(filtered_confidences) > 0:
        print(f"\n🎯 Detected Objects:")
        class_counts = {}
        for cls_id, conf in zip(filtered_class_ids, filtered_confidences):
            class_name = VEHICLE_CLASSES.get(cls_id, f"unknown_{cls_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if conf > 0.4:  # Show high confidence detections
                print(f"   - {class_name}: {conf:.3f}")
        
        print(f"\n📋 Summary by Class:")
        for class_name, count in sorted(class_counts.items()):
            print(f"   - {class_name}: {count} objects")
    else:
        print("   ❌ No objects detected above threshold")
    
    return filtered_confidences

def test_model_on_npy_files(model_path, npy_dir, num_files=5):
    """Test model on .npy files from shared_data"""
    print("=" * 60)
    print("🧪 Testing ONNX Model Detection")
    print("=" * 60)
    
    # Load model
    print(f"\n🧠 Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return
    
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"✅ Model loaded successfully")
    
    # Find .npy files
    npy_files = glob.glob(os.path.join(npy_dir, "**/*.npy"), recursive=True)
    
    if not npy_files:
        print(f"\n❌ No .npy files found in {npy_dir}")
        return
    
    print(f"\n📁 Found {len(npy_files)} .npy files")
    print(f"🔍 Testing on first {min(num_files, len(npy_files))} files...\n")
    
    total_detections = 0
    
    # Test on multiple files
    for idx, npy_file in enumerate(npy_files[:num_files]):
        print(f"\n{'='*60}")
        print(f"📦 File {idx+1}/{min(num_files, len(npy_files))}: {os.path.basename(npy_file)}")
        print(f"{'='*60}")
        
        try:
            # Load batch
            frames_batch = np.load(npy_file)
            print(f"   Shape: {frames_batch.shape}")
            print(f"   Dtype: {frames_batch.dtype}")
            print(f"   Value range: [{frames_batch.min()}, {frames_batch.max()}]")
            
            # Test on first 3 frames
            num_frames = min(3, len(frames_batch))
            print(f"\n   Testing on first {num_frames} frames:")
            
            for frame_idx in range(num_frames):
                frame = frames_batch[frame_idx]
                print(f"\n   --- Frame {frame_idx + 1} ---")
                
                # Preprocess
                input_tensor = preprocess_frame(frame)
                
                # Inference
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: input_tensor})
                
                # Postprocess
                detections = postprocess_simple(outputs, conf_thresh=0.25)
                total_detections += len(detections)
            
        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 Final Results:")
    print(f"   Total files tested: {min(num_files, len(npy_files))}")
    print(f"   Total detections: {total_detections}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Paths
    MODEL_PATH = r"d:\Truck-classification\Truck-Deploy\truck_classification.onnx"
    NPY_DIR = r"d:\Truck-classification\shared_data\VIDEO_TEST"
    
    # Run test
    test_model_on_npy_files(MODEL_PATH, NPY_DIR, num_files=3)
