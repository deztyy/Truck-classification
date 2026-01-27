-- Vehicle Classes Table
CREATE TABLE IF NOT EXISTS vehicle_classes (
    class_id INT PRIMARY KEY,
    class_name VARCHAR(50) NOT NULL,
    entry_fee DECIMAL(10,2),
    xray_fee DECIMAL(10,2),
    total_fee DECIMAL(10,2)
);

-- Vehicle Transactions Table
CREATE TABLE IF NOT EXISTS vehicle_transactions (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(100),
    track_id VARCHAR(100),
    class_id INT REFERENCES vehicle_classes(class_id),
    total_fee DECIMAL(10,2) DEFAULT 0.00,
    time_stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    img_path TEXT,
    confidence FLOAT
);

-- Insert Vehicle Classes Reference Data
INSERT INTO vehicle_classes (class_id, class_name, entry_fee, xray_fee, total_fee) VALUES
(0, 'car', 0.00, 0.00, 0.00),
(1, 'other', 0.00, 0.00, 0.00),
(2, 'other_truck', 100.00, 50.00, 150.00),
(3, 'pickup_truck', 0.00, 0.00, 0.00),
(4, 'truck_20_back', 100.00, 250.00, 350.00),
(5, 'truck_20_front', 100.00, 250.00, 350.00),
(6, 'truck_20x2', 100.00, 500.00, 600.00),
(7, 'truck_40', 100.00, 350.00, 450.00),
(8, 'truck_roro', 100.00, 50.00, 150.00),
(9, 'truck_tail', 100.00, 50.00, 150.00),
(10, 'motorcycle', 0.00, 0.00, 0.00),
(11, 'truck_head', 100.00, 50.00, 150.00)
ON CONFLICT (class_id) DO NOTHING;

-- Create Indexes for Better Performance
CREATE INDEX IF NOT EXISTS idx_vehicle_transactions_camera_id ON vehicle_transactions(camera_id);
CREATE INDEX IF NOT EXISTS idx_vehicle_transactions_time_stamp ON vehicle_transactions(time_stamp);
CREATE INDEX IF NOT EXISTS idx_vehicle_transactions_class_id ON vehicle_transactions(class_id);
