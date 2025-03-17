#!/usr/bin/env python3
"""
DeGirum Hailo 8 Fire and Smoke Detection System (for RTSP Stream)
Connects directly to RTSP camera, processes images, and performs fire and smoke detection using Hailo 8.
Includes MQTT and Home Assistant integration.
"""

import os
import sys
import cv2
import time
import json
import numpy as np
import logging
import threading
import queue
import requests
import tempfile
import base64
import paho.mqtt.client as mqtt
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fire_smoke_detection.log')
    ]
)
logger = logging.getLogger("hailo_fire_smoke_detection")

# Import DeGirum API
import degirum as dg

# RTSP Configuration
RTSP_URL = "rtsp://username:password@camera_ip:554/stream1"

# Configuration
CONFIG = {
    "detection_threshold": 0.5,  # Detection threshold
    "frame_skip": 2,  # How many frames to skip (for performance)
    "display_output": False,  # Show output - disabled by default
    "save_detections": True,  # Save detection images
    "alert_mode": True,  # Enable alarm mode
}

# Home Assistant Configuration
HOME_ASSISTANT_CONFIG = {
    "url": "http://homeassistant_ip:8123",
    "token": "your_home_assistant_token",
    "sensor_name": "sensor.hailo_fire_detection",
    "update_interval": 5  # How often to update Home Assistant in seconds
}

# MQTT Configuration
MQTT_CONFIG = {
    "enabled": True,
    "host": "mqtt_broker_ip",
    "port": 1883,
    "user": "mqtt_username",
    "password": "mqtt_password",
    "topic_prefix": "hailo/fire",
    "state_topic": "hailo/fire/state",
    "image_topic": "hailo/fire/image",
    "availability_topic": "hailo/fire/availability",
    "update_interval": 2  # How often to update MQTT in seconds
}

# DeGirum Configuration
model_path = os.environ.get("MODEL_PATH", "/path/to/models/yolov8n_relu6_fire_smoke--640x640_quant_hailort_hailo8_1")
zoo_url = "/path/to/models"
inference_host_address = "@local"
token = ''
model_name = "yolov8n_relu6_fire_smoke--640x640_quant_hailort_hailo8_1"


# Create directories
DETECTION_DIR = "detection_images"
DEBUG_IMAGES_DIR = "debug_images"
os.makedirs(DETECTION_DIR, exist_ok=True)
os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)

# Initialize MQTT client
mqtt_client = None
if MQTT_CONFIG["enabled"]:
    try:
        mqtt_client = mqtt.Client(client_id="hailo-fire-detection")
        mqtt_client.username_pw_set(MQTT_CONFIG["user"], MQTT_CONFIG["password"])
        mqtt_client.will_set(MQTT_CONFIG["availability_topic"], "offline", qos=1, retain=True)
        mqtt_client.connect(MQTT_CONFIG["host"], MQTT_CONFIG["port"], 60)
        mqtt_client.loop_start()
        # Publish availability notification
        mqtt_client.publish(MQTT_CONFIG["availability_topic"], "online", qos=1, retain=True)
        logger.info(f"Connected to MQTT server: {MQTT_CONFIG['host']}:{MQTT_CONFIG['port']}")
        
        # Publish configuration for Home Assistant MQTT auto-discovery
        discovery_config = {
            "name": "Hailo Fire Detection",
            "device_class": "smoke",
            "state_topic": MQTT_CONFIG["state_topic"],
            "availability_topic": MQTT_CONFIG["availability_topic"],
            "payload_available": "online",
            "payload_not_available": "offline",
            "value_template": "{{ value_json.state }}",
            "json_attributes_topic": MQTT_CONFIG["state_topic"],
            "unique_id": "hailo_fire_detection",
            "device": {
                "identifiers": ["hailo_fire_detection"],
                "name": "Hailo Fire Detection",
                "model": "Hailo 8",
                "manufacturer": "DeGirum"
            }
        }
        mqtt_client.publish("homeassistant/binary_sensor/hailo_fire/config", 
                           json.dumps(discovery_config), qos=1, retain=True)
        
        # MQTT discovery configuration for camera
        camera_discovery_config = {
            "name": "Hailo Fire Detection Camera",
            "topic": MQTT_CONFIG["image_topic"],
            "unique_id": "hailo_fire_detection_camera",
            "device": {
                "identifiers": ["hailo_fire_detection"],
                "name": "Hailo Fire Detection",
                "model": "Hailo 8",
                "manufacturer": "DeGirum"
            }
        }
        mqtt_client.publish("homeassistant/camera/hailo_fire/config", 
                           json.dumps(camera_discovery_config), qos=1, retain=True)
        
    except Exception as e:
        logger.error(f"MQTT connection error: {str(e)}")
        mqtt_client = None


class FireSmokeDetector:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.running = False
        self.frame_count = 0
        self.detection_count = 0
        self.last_alert_time = time.time() - 100  # To trigger alert immediately at the start
        self.last_mqtt_update_time = time.time() - 100  # For MQTT updates
        
        # Detection status
        self.current_detections = {
            "fire": False,
            "smoke": False,
            "last_fire_time": None,
            "last_smoke_time": None,
            "detection_count": 0,
            "fire_confidence": 0.0,
            "smoke_confidence": 0.0
        }
        
        # Last processed frame
        self.last_processed_frame = None
        
        # Load DeGirum model
        self.load_model()
        
        # Class labels
        self.class_names = ['fire', 'smoke']
    
    def update_mqtt(self, processed_frame=None, force=False):
        """Update status via MQTT"""
        global mqtt_client
        
        if not MQTT_CONFIG["enabled"] or mqtt_client is None:
            return
            
        try:
            current_time = time.time()
            # Update every update_interval seconds or when force=True
            if force or (current_time - self.last_mqtt_update_time > MQTT_CONFIG["update_interval"]):
                self.last_mqtt_update_time = current_time
                
                # Prepare status data
                now = datetime.now().isoformat()
                state_data = {
                    "state": "ON" if self.current_detections["fire"] or self.current_detections["smoke"] else "OFF",
                    "fire_detected": self.current_detections["fire"],
                    "smoke_detected": self.current_detections["smoke"],
                    "last_fire_time": self.current_detections["last_fire_time"],
                    "last_smoke_time": self.current_detections["last_smoke_time"],
                    "detection_count": self.current_detections["detection_count"],
                    "fire_confidence": self.current_detections["fire_confidence"],
                    "smoke_confidence": self.current_detections["smoke_confidence"],
                    "last_updated": now
                }
                
                # Publish to status topic
                mqtt_client.publish(MQTT_CONFIG["state_topic"], json.dumps(state_data), qos=1, retain=True)
                
                # Send image in detection state
                if processed_frame is not None:
                    if self.current_detections["fire"] or self.current_detections["smoke"]:
                        # Resize and reduce quality of image (for faster transmission)
                        max_width = 640
                        height, width = processed_frame.shape[:2]
                        
                        if width > max_width:
                            ratio = max_width / width
                            new_height = int(height * ratio)
                            small_frame = cv2.resize(processed_frame, (max_width, new_height))
                        else:
                            small_frame = processed_frame
                            
                        # Encode as JPEG
                        success, jpg_buffer = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        
                        if success:
                            # Base64 encode
                            jpg_as_text = base64.b64encode(jpg_buffer).decode('utf-8')
                            # Publish to image topic
                            mqtt_client.publish(MQTT_CONFIG["image_topic"], jpg_as_text, qos=0, retain=True)
                            logger.debug("Detection image sent via MQTT")
                
                logger.debug("MQTT status updated")
                
        except Exception as e:
            logger.error(f"MQTT update error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    def load_model(self):
        """Load DeGirum Hailo 8 model"""
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Use dg.load_model as shown in the example code
            self.model = dg.load_model(
                model_name=model_name,
                inference_host_address=inference_host_address,
                zoo_url=zoo_url,
                token=token
            )
            
            logger.info(f"Model loaded successfully.")
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            sys.exit(1)

    def update_home_assistant(self):
        """Update Home Assistant sensor"""
        try:
            # Create sensor data
            now = datetime.now().isoformat()
            sensor_data = {
                "state": "on" if self.current_detections["fire"] or self.current_detections["smoke"] else "off",
                "attributes": {
                    "friendly_name": "Hailo Fire Detection",
                    "device_class": "fire",
                    "fire_detected": self.current_detections["fire"],
                    "smoke_detected": self.current_detections["smoke"],
                    "last_fire_time": self.current_detections["last_fire_time"],
                    "last_smoke_time": self.current_detections["last_smoke_time"],
                    "detection_count": self.current_detections["detection_count"],
                    "fire_confidence": self.current_detections["fire_confidence"],
                    "smoke_confidence": self.current_detections["smoke_confidence"],
                    "last_updated": now
                }
            }
            
            # Send to Home Assistant API
            headers = {
                "Authorization": f"Bearer {HOME_ASSISTANT_CONFIG['token']}",
                "Content-Type": "application/json"
            }
            
            api_url = f"{HOME_ASSISTANT_CONFIG['url']}/api/states/{HOME_ASSISTANT_CONFIG['sensor_name']}"
            
            response = requests.post(api_url, headers=headers, json=sensor_data)
            
            if response.status_code == 200 or response.status_code == 201:
                logger.info(f"Home Assistant sensor updated: {HOME_ASSISTANT_CONFIG['sensor_name']}")
            else:
                logger.error(f"Failed to update Home Assistant sensor. Status code: {response.status_code}, Response: {response.text}")
                
        except Exception as e:
            logger.error(f"Home Assistant update error: {str(e)}")
            
    def process_frame(self, frame):
        """Process a single frame and return results"""
        try:
            # Preprocessing and inference start
            start_time = time.time()
            original_size = (frame.shape[0], frame.shape[1])
            
            # Resize to the model's required size (640x640)
            resized_frame = cv2.resize(frame, (640, 640))
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
            
            # Save the image with high quality
            cv2.imwrite(temp_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Inference - call the model directly as a function
            result = self.model(temp_path)
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
                
            inference_time = time.time() - start_time
            
            # Update detection states
            fire_detected = False
            smoke_detected = False
            fire_confidence = 0.0
            smoke_confidence = 0.0
            
            detections = []
            
            # Process results
            if hasattr(result, 'results') and isinstance(result.results, list):
                logger.debug(f"Detection results: {len(result.results)}")
                
                for detection in result.results:
                    # Check 'bbox' field
                    if 'bbox' in detection:
                        bbox = detection['bbox']
                        if len(bbox) == 4:
                            # Normalize coordinates from the model (640x640) to original frame size
                            x1, y1, x2, y2 = map(float, bbox)
                            
                            # Convert to original size if model outputs normalized coordinates (0-1)
                            if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                                x1 = int(x1 * original_size[1])
                                y1 = int(y1 * original_size[0])
                                x2 = int(x2 * original_size[1])
                                y2 = int(y2 * original_size[0])
                            else:
                                # Scale 640x640 coordinates to original size
                                x1 = int(x1 * original_size[1] / 640)
                                y1 = int(y1 * original_size[0] / 640)
                                x2 = int(x2 * original_size[1] / 640)
                                y2 = int(y2 * original_size[0] / 640)
                            
                            # Get class ID and score
                            class_id = detection.get('class_id', 0)
                            score = detection.get('score', 0)
                            class_name = detection.get('class_name', '')
                            
                            if score > CONFIG["detection_threshold"]:
                                # DeGirum returns class_name directly
                                if class_name == "fire":
                                    fire_detected = True
                                    fire_confidence = max(fire_confidence, score)
                                elif class_name == "smoke":
                                    smoke_detected = True  
                                    smoke_confidence = max(smoke_confidence, score)
                                
                                detections.append({
                                    "box": [x1, y1, x2, y2],
                                    "score": float(score),
                                    "class_id": int(class_id),
                                    "class_name": class_name
                                })
            
            # Update detection status
            now = datetime.now().isoformat()
            
            # Update detection status
            if fire_detected:
                self.current_detections["fire"] = True
                self.current_detections["last_fire_time"] = now
                self.current_detections["fire_confidence"] = fire_confidence
                
            if smoke_detected:
                self.current_detections["smoke"] = True
                self.current_detections["last_smoke_time"] = now
                self.current_detections["smoke_confidence"] = smoke_confidence
                
            if fire_detected or smoke_detected:
                self.current_detections["detection_count"] += 1
            
            # Save results
            processed_frame = frame.copy()
            if detections:
                processed_frame = self.draw_detections(processed_frame, detections)
                self.detection_count += 1
                
                # Save last processed frame
                self.last_processed_frame = processed_frame.copy()
                
                # Save detection
                if CONFIG["save_detections"]:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    detection_filename = f"{DETECTION_DIR}/detection_{timestamp}.jpg"
                    cv2.imwrite(detection_filename, processed_frame)
                    logger.info(f"Detection saved: {detection_filename}")
                
                # Send alert
                if CONFIG["alert_mode"] and time.time() - self.last_alert_time > 10:  # Alert every 10 seconds
                    self.last_alert_time = time.time()
                    logger.warning(f"ALERT: {len(detections)} fire/smoke detected!")
                    
                    # Update Home Assistant
                    self.update_home_assistant()
                    
                    # Update MQTT and send image
                    self.update_mqtt(processed_frame, force=True)
                else:
                    # Normal MQTT update
                    self.update_mqtt(processed_frame)
            else:
                # Update MQTT even if no detection
                self.update_mqtt()
            
            # Calculate FPS
            fps = 1.0 / inference_time
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            return processed_frame, detections, fps
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return frame, [], 0
    
    def draw_detections(self, frame, detections):
        """Draw detections on the image"""
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            class_id = det["class_id"]
            score = det["score"]
            class_name = det["class_name"]
            
            # Select color based on class
            color = (0, 0, 255) if class_name == "fire" else (0, 165, 255)  # Red: fire, Orange: smoke
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        return frame
    
    def capture_thread(self):
        """Capture video from RTSP stream"""
        logger.info(f"Starting RTSP URL connection: {RTSP_URL}")
        cap = cv2.VideoCapture(RTSP_URL)
        
        if not cap.isOpened():
            logger.error(f"Failed to open RTSP stream: {RTSP_URL}")
            self.running = False
            return
        
        logger.info("Video capture started")
        
        frame_count = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame not captured, trying to reconnect...")
                time.sleep(1)
                cap = cv2.VideoCapture(RTSP_URL)
                continue
            
            frame_count += 1
            # Skip frames for performance
            if frame_count % CONFIG["frame_skip"] != 0:
                continue
                
            # Add frame to queue, drop oldest if full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put(frame)
            except:
                pass
        
        cap.release()
        logger.info("Video capture stopped")
    
    def processing_thread(self):
        """Process frames from the queue"""
        logger.info("Processing started")
        
        while self.running:
            try:
                # Get next frame from queue
                frame = self.frame_queue.get(timeout=1)
                
                # Process frame
                processed_frame, detections, fps = self.process_frame(frame)
                
                # Add to result queue
                self.result_queue.put((processed_frame, detections, fps))
                
                self.frame_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
        
        logger.info("Processing stopped")
    
    def display_thread(self):
        """Display processed frames"""
        logger.info("Display started")
        
        last_log_time = time.time()
        frames_since_log = 0
        
        try:
            # Check for display before creating window
            if CONFIG["display_output"]:
                if not os.environ.get('DISPLAY'):
                    logger.warning("DISPLAY environment variable not found, disabling display")
                    CONFIG["display_output"] = False
        except Exception as e:
            logger.warning(f"Display check failed: {str(e)}, disabling display")
            CONFIG["display_output"] = False
        
        while self.running:
            try:
                # Get processed frame from result queue
                processed_frame, detections, fps = self.result_queue.get(timeout=1)
                
                # Show frame
                if CONFIG["display_output"]:
                    try:
                        cv2.imshow("DeGirum Hailo 8 - Fire and Smoke Detection", processed_frame)
                        
                        # Exit if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.running = False
                            break
                    except Exception as e:
                        logger.error(f"Display error: {str(e)}")
                        CONFIG["display_output"] = False
                        logger.warning("Display disabled")
                
                frames_since_log += 1
                
                # Log statistics every 10 seconds
                if time.time() - last_log_time > 10:
                    fps_avg = frames_since_log / (time.time() - last_log_time)
                    logger.info(f"Statistics - FPS: {fps_avg:.2f}, Processed frames: {self.frame_count}, Detections: {self.detection_count}")
                    last_log_time = time.time()
                    frames_since_log = 0
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Display error: {str(e)}")
        
        # Close display window
        if CONFIG["display_output"]:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        logger.info("Display stopped")
    
    def home_assistant_thread(self):
        """Periodically update Home Assistant"""
        logger.info("Home Assistant update thread started")
        
        while self.running:
            try:
                # Update Home Assistant at specified intervals
                time.sleep(HOME_ASSISTANT_CONFIG["update_interval"])
                
                # Only update if detection (to avoid unnecessary API calls)
                if self.current_detections["fire"] or self.current_detections["smoke"]:
                    self.update_home_assistant()
                    
                # Reset status if no new detection for 30 seconds
                if (self.current_detections["fire"] and 
                    self.current_detections["last_fire_time"] and
                    (datetime.now() - datetime.fromisoformat(self.current_detections["last_fire_time"])).total_seconds() > 30):
                    self.current_detections["fire"] = False
                    self.current_detections["fire_confidence"] = 0.0
                    # Update Home Assistant
                    self.update_home_assistant()
                    
                if (self.current_detections["smoke"] and 
                    self.current_detections["last_smoke_time"] and
                    (datetime.now() - datetime.fromisoformat(self.current_detections["last_smoke_time"])).total_seconds() > 30):
                    self.current_detections["smoke"] = False
                    self.current_detections["smoke_confidence"] = 0.0
                    # Update Home Assistant
                    self.update_home_assistant()
                    
            except Exception as e:
                logger.error(f"Home Assistant update error: {str(e)}")
                
        logger.info("Home Assistant update thread stopped")
    
    def mqtt_thread(self):
        """Periodically update MQTT status"""
        logger.info("MQTT update thread started")
        
        while self.running:
            try:
                # Update MQTT at specified intervals
                time.sleep(MQTT_CONFIG["update_interval"])
                
                # Send status with last processed frame if available
                if self.last_processed_frame is not None:
                    self.update_mqtt(self.last_processed_frame)
                else:
                    self.update_mqtt()
                    
          
                # Reset status if no new detection for 30 seconds
                if (self.current_detections["fire"] and 
                    self.current_detections["last_fire_time"] and
                    (datetime.now() - datetime.fromisoformat(self.current_detections["last_fire_time"])).total_seconds() > 30):
                    self.current_detections["fire"] = False
                    self.current_detections["fire_confidence"] = 0.0
                    # Update MQTT
                    self.update_mqtt(force=True)
                    
                if (self.current_detections["smoke"] and 
                    self.current_detections["last_smoke_time"] and
                    (datetime.now() - datetime.fromisoformat(self.current_detections["last_smoke_time"])).total_seconds() > 30):
                    self.current_detections["smoke"] = False
                    self.current_detections["smoke_confidence"] = 0.0
                    # Update MQTT
                    self.update_mqtt(force=True)
                    
            except Exception as e:
                logger.error(f"MQTT update error: {str(e)}")
                
        logger.info("MQTT update thread stopped")
        
        # Signal offline status on exit
        if MQTT_CONFIG["enabled"] and mqtt_client is not None:
            try:
                mqtt_client.publish(MQTT_CONFIG["availability_topic"], "offline", qos=1, retain=True)
                mqtt_client.disconnect()
            except:
                pass
    
    def start(self):
        """Start all threads"""
        self.running = True
        
        # Create threads
        self.capture_thread_obj = threading.Thread(target=self.capture_thread)
        self.processing_thread_obj = threading.Thread(target=self.processing_thread)
        self.display_thread_obj = threading.Thread(target=self.display_thread)
        self.home_assistant_thread_obj = threading.Thread(target=self.home_assistant_thread)
        
        # Add MQTT thread
        if MQTT_CONFIG["enabled"] and mqtt_client is not None:
            self.mqtt_thread_obj = threading.Thread(target=self.mqtt_thread)
        else:
            self.mqtt_thread_obj = None
        
        # Start threads
        self.capture_thread_obj.start()
        self.processing_thread_obj.start()
        self.display_thread_obj.start()
        self.home_assistant_thread_obj.start()
        
        # Start MQTT thread
        if self.mqtt_thread_obj is not None:
            self.mqtt_thread_obj.start()
        
        logger.info("All threads started")
        
        try:
            # Keep main thread running
            while self.running:
                time.sleep(0.1)
                
                # User can terminate with CTRL+C
                active_threads = [self.capture_thread_obj.is_alive(), 
                                 self.processing_thread_obj.is_alive(),
                                 self.display_thread_obj.is_alive(),
                                 self.home_assistant_thread_obj.is_alive()]
                
                if self.mqtt_thread_obj is not None:
                    active_threads.append(self.mqtt_thread_obj.is_alive())
                
                if not all(active_threads):
                    logger.warning("A thread stopped, shutting down...")
                    self.running = False
                    break
                    
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            self.running = False
        
        # Wait for threads to finish
        self.capture_thread_obj.join()
        self.processing_thread_obj.join()
        self.display_thread_obj.join()
        self.home_assistant_thread_obj.join()
        
        # Wait for MQTT thread
        if self.mqtt_thread_obj is not None:
            self.mqtt_thread_obj.join()
        
        logger.info("Application stopped")


def main():
    logger.info("Starting DeGirum Hailo 8 Fire and Smoke Detection System...")
    
    # Create initial Home Assistant sensor
    try:
        headers = {
            "Authorization": f"Bearer {HOME_ASSISTANT_CONFIG['token']}",
            "Content-Type": "application/json"
        }
        
        sensor_data = {
            "state": "off",
            "attributes": {
                "friendly_name": "Hailo Fire Detection",
                "device_class": "fire",
                "fire_detected": False,
                "smoke_detected": False,
                "last_updated": datetime.now().isoformat()
            }
        }
        
        api_url = f"{HOME_ASSISTANT_CONFIG['url']}/api/states/{HOME_ASSISTANT_CONFIG['sensor_name']}"
        
        response = requests.post(api_url, headers=headers, json=sensor_data)
        
        if response.status_code == 200 or response.status_code == 201:
            logger.info(f"Home Assistant sensor created: {HOME_ASSISTANT_CONFIG['sensor_name']}")
        else:
            logger.error(f"Could not create Home Assistant sensor. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logger.error(f"Home Assistant initialization error: {str(e)}")
    
    # Send initial MQTT status
    if MQTT_CONFIG["enabled"] and mqtt_client is not None:
        try:
            initial_state = {
                "state": "OFF",
                "fire_detected": False,
                "smoke_detected": False,
                "detection_count": 0,
                "last_updated": datetime.now().isoformat()
            }
            mqtt_client.publish(MQTT_CONFIG["state_topic"], json.dumps(initial_state), qos=1, retain=True)
            logger.info("MQTT initial status sent")
        except Exception as e:
            logger.error(f"MQTT initial status error: {str(e)}")
    
    detector = FireSmokeDetector()
    detector.start()


if __name__ == "__main__":
    main()
