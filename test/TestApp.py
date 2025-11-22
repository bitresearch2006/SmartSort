#!/usr/bin/env python3
"""
Simplified SmartSort TestApp
Features:
1. Pass image file or capture from camera
2. Choose local or remote processing
3. Show the input image
"""

import os
import sys
import base64
import json
import requests
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# Add main directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))

class SmartSortTester:
    def __init__(self):
        self.image_path = None
        self.image_b64 = None
        self.mode = 'local'  # 'local' or 'remote'
        
    def capture_from_camera(self, camera_index=0):
        """Capture image from system camera"""
        print("📷 Initializing camera...")
        
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                print(f"❌ Error: Could not open camera {camera_index}")
                print("💡 Try checking if camera is available or being used by another app")
                return False
            
            # Set camera properties for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("📸 Camera ready! Press SPACE to capture, ESC to cancel")
            print("🔍 Position your biomedical waste item in front of the camera")
            
            while True:
                # Read frame from camera
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Display the frame
                cv2.imshow('SmartSort Camera - Press SPACE to capture, ESC to cancel', frame)
                
                # Wait for key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == 32:  # SPACE key
                    # Capture the image
                    captured_image = frame.copy()
                    
                    # Save the captured image temporarily
                    temp_path = "captured_image.jpg"
                    cv2.imwrite(temp_path, captured_image)
                    
                    print(f"✅ Image captured and saved as {temp_path}")
                    
                    # Clean up
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    # Load the captured image using existing method
                    success = self.load_image_file(temp_path)
                    
                    if success:
                        print("� Camera capture successful!")
                        return True
                    else:
                        print("❌ Error processing captured image")
                        return False
                        
                elif key == 27:  # ESC key
                    print("📷 Camera capture cancelled")
                    break
            
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            return False
            
        except Exception as e:
            print(f"❌ Error during camera capture: {e}")
            print("💡 Make sure OpenCV is installed: pip install opencv-python")
            return False
    
    def list_available_cameras(self):
        """List available cameras in the system"""
        print("🔍 Scanning for available cameras...")
        available_cameras = []
        
        # Test cameras 0-5 (usually sufficient)
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"✅ Camera {i}: Available")
                else:
                    print(f"❌ Camera {i}: Cannot read frames")
                cap.release()
            else:
                print(f"❌ Camera {i}: Not available")
        
        if not available_cameras:
            print("❌ No cameras found")
        else:
            print(f"📷 Found {len(available_cameras)} available camera(s): {available_cameras}")
            
        return available_cameras
        
    def load_image_file(self, file_path):
        """Load image from file path"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ Error: Image file not found: {file_path}")
                return False
                
            # Validate it's an image file
            try:
                img = Image.open(file_path)
                img.verify()  # Verify it's a valid image
            except Exception:
                print(f"❌ Error: Invalid image file: {file_path}")
                return False
                
            self.image_path = file_path
            
            # Convert to base64 for processing
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            self.image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            print(f"✅ Image loaded successfully: {file_path}")
            print(f"📊 Base64 size: {len(self.image_b64)} characters")
            
            # Always show the image when loaded
            self.show_image()
            return True
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return False
            
    def show_image(self):
        """Display the input image"""
        if not self.image_path:
            print("❌ No image loaded to display")
            return
            
        try:
            # Load and display image
            img = mpimg.imread(self.image_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Input Image: {os.path.basename(self.image_path)}")
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error displaying image: {e}")
            
    def test_local(self):
        """Test with local model"""
        print("\\n🖥️  Testing with LOCAL model...")
        print("-" * 40)
        
        try:
            # Import handler function
            from handler import handle
            
            # Prepare request data
            request_data = {"image_b64": self.image_b64}
            request_json = json.dumps(request_data)
            
            # Call local handler
            result = handle(request_json)
            
            # Parse and display result
            try:
                result_dict = json.loads(result)
                self.display_result(result_dict, "LOCAL")
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON response from local handler")
                print(f"Raw response: {result}")
                
        except ImportError:
            print("❌ Error: Could not import handler module")
            print("Make sure handler.py is in the main/ directory")
        except Exception as e:
            print(f"❌ Error during local testing: {e}")
            
    def test_remote(self, endpoint="http://localhost:8080/function/SmartSort"):
        """Test with remote OpenFaaS endpoint"""
        print("\\n🌐 Testing with REMOTE endpoint...")
        print("-" * 40)
        print(f"🔗 Endpoint: {endpoint}")
        
        try:
            # Prepare request data
            request_data = {"image_b64": self.image_b64}
            
            # Send request to remote endpoint
            response = requests.post(
                endpoint,
                json=request_data,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"📡 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result_dict = response.json()
                    self.display_result(result_dict, "REMOTE")
                except json.JSONDecodeError:
                    print("❌ Error: Invalid JSON response from remote endpoint")
                    print(f"Raw response: {response.text}")
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to remote endpoint")
            print("Make sure OpenFaaS is running and the function is deployed")
        except requests.exceptions.Timeout:
            print("❌ Error: Request timeout (30s)")
        except Exception as e:
            print(f"❌ Error during remote testing: {e}")
            
    def display_result(self, result_dict, mode):
        """Display classification results in a formatted way"""
        print(f"\\n🎯 {mode} CLASSIFICATION RESULT:")
        print("=" * 50)
        
        if result_dict.get("status") == "success":
            classification = result_dict.get("classification", "Unknown")
            confidence = result_dict.get("confidence_percent", 0)
            
            print(f"📋 Classification: {classification}")
            print(f"🎯 Confidence: {confidence:.2f}%")
            
            # Display probability breakdown
            probabilities = result_dict.get("probability_breakdown", {})
            if probabilities:
                print(f"\\n📊 Probability Breakdown:")
                print("-" * 30)
                
                # Sort by probability (highest first)
                sorted_probs = sorted(probabilities.items(), 
                                    key=lambda x: x[1], reverse=True)
                
                for category, prob in sorted_probs:
                    bar_length = int(prob / 100 * 20)  # Scale to 20 chars
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"{category:35} {bar} {prob:5.1f}%")
        else:
            print(f"❌ Error: {result_dict.get('message', 'Unknown error')}")
            
    def run_interactive(self):
        """Run interactive mode"""
        print("🚀 SmartSort - Interactive Mode")
        print("=" * 50)
        
        while True:
            print("\\n📋 Options:")
            print("1. Load image file")
            print("2. Capture from camera")
            print("3. List available cameras")
            print("4. Show current image")
            print("5. Test local")
            print("6. Test remote")
            print("7. Exit")
            
            choice = input("\\nEnter choice (1-7): ").strip()
            
            if choice == '1':
                file_path = input("Enter image file path: ").strip().strip('"')
                self.load_image_file(file_path)
                
            elif choice == '2':
                cameras = self.list_available_cameras()
                if cameras:
                    if len(cameras) == 1:
                        camera_idx = cameras[0]
                        print(f"Using camera {camera_idx}")
                    else:
                        try:
                            camera_idx = int(input(f"Select camera {cameras}: "))
                            if camera_idx not in cameras:
                                print(f"❌ Invalid camera selection. Using camera {cameras[0]}")
                                camera_idx = cameras[0]
                        except ValueError:
                            print(f"❌ Invalid input. Using camera {cameras[0]}")
                            camera_idx = cameras[0]
                    
                    self.capture_from_camera(camera_idx)
                else:
                    print("❌ No cameras available")
                
            elif choice == '3':
                self.list_available_cameras()
                
            elif choice == '4':
                self.show_image()
                
            elif choice == '5':
                if self.image_b64:
                    self.test_local()
                else:
                    print("❌ Please load an image first")
                    
            elif choice == '6':
                if self.image_b64:
                    endpoint = input("Enter endpoint (or press Enter for default): ").strip()
                    if not endpoint:
                        endpoint = "http://localhost:8080/function/SmartSort"
                    self.test_remote(endpoint)
                else:
                    print("❌ Please load an image first")
                    
            elif choice == '7':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-7.")

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(
        description="SmartSort - Biomedical Waste Classification Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python TestApp.py                                    # Interactive mode
  python TestApp.py --image mask.jpg --mode local     # Test local with image file
  python TestApp.py --camera --mode local             # Capture from camera and test local
  python TestApp.py --camera --camera-index 1         # Use camera 1 instead of default camera 0
  python TestApp.py --list-cameras                     # List available cameras
  python TestApp.py --image mask.jpg --mode remote    # Test remote with image file
        """
    )
    
    parser.add_argument('--image', '-i', 
                       help='Path to image file')
    parser.add_argument('--camera', '-c',
                       action='store_true',
                       help='Capture image from camera')
    parser.add_argument('--camera-index',
                       type=int,
                       default=0,
                       help='Camera index to use (default: 0)')
    parser.add_argument('--mode', '-m', 
                       choices=['local', 'remote'], 
                       default='local',
                       help='Processing mode (default: local)')
    parser.add_argument('--endpoint', '-e',
                       default='http://localhost:8080/function/SmartSort',
                       help='Remote endpoint URL (default: OpenFaaS local)')
    parser.add_argument('--list-cameras',
                       action='store_true',
                       help='List available cameras and exit')
    parser.add_argument('--interactive',
                       action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = SmartSortTester()
    
    # Handle list cameras command
    if args.list_cameras:
        tester.list_available_cameras()
        return
    
    # If no arguments provided, run interactive mode
    if len(sys.argv) == 1 or args.interactive:
        tester.run_interactive()
        return
    
    # Command line mode
    print("🚀 SmartSort - Command Line Mode")
    print("=" * 50)
    
    # Handle image input
    image_loaded = False
    
    if args.image:
        # Load image from file
        image_loaded = tester.load_image_file(args.image)
    elif args.camera:
        # Capture from camera
        image_loaded = tester.capture_from_camera(args.camera_index)
    else:
        print("❌ Please provide either --image or --camera")
        parser.print_help()
        return
    
    if not image_loaded:
        print("❌ Failed to load image")
        return
        
    # Run classification
    if args.mode == 'local':
        tester.test_local()
    elif args.mode == 'remote':
        tester.test_remote(args.endpoint)

if __name__ == "__main__":
    main()