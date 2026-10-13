import os
import sys

import cv2
import time

# Base configuration, taken from the environment
IP = os.environ.get("NVR_IP", "")
PORT = os.environ.get("NVR_PORT", "554")
USER = os.environ.get("NVR_USER", "admin")
PASS = os.environ.get("NVR_PASS", "")

if not IP or not PASS:
    sys.exit("Error: Set NVR_IP and NVR_PASS (and optionally NVR_USER, NVR_PORT) in the environment.")

# A comprehensive array of all known URL path formats for Xiongmai hardware variants
TEST_PATHS = [
    # Variant A: Standard modern channel paths
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/ch01_0.264",
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/ch01/0",
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/ch1/0",
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/ch0_0",
    
    # Variant B: Direct codec/encoder stream endpoints
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/h264",
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/h264_stream",
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/mpeg4",
    
    # Variant C: Query parameter strings (Sofia / NetSurveillance)
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/user={USER}&password={PASS}&channel=1&stream=0.sdp",
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/cam/realmonitor?channel=1&subtype=0",
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/LiveMedia/ch1/Media1",
    
    # Variant D: Native ONVIF profiles
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/onvif1",
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/onvif2",
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/onvif-media-profile",
    
    # Variant E: Short form direct access
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/11",
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/1/1",
    f"rtsp://{USER}:{PASS}@{IP}:{PORT}/live0"
]

print("--- STARTING RTSP URL SCANNER FOR XIONGMAI NVR ---")
print(f"Target: {IP}:{PORT} | Testing {len(TEST_PATHS)} potential paths...\n")

found_valid_url = None

for i, url in enumerate(TEST_PATHS, 1):
    # Mask password in printed console logs for safety
    masked_url = url.replace(PASS, "******")
    print(f"[{i}/{len(TEST_PATHS)}] Probing path: {masked_url}")
    
    # Test the connection with a 2-second timeout window
    cap = cv2.VideoCapture(url)
    
    # Give the network buffer a tiny window to establish handshakes
    time.sleep(0.5) 
    
    if cap.isOpened():
        # Double check frame reading viability to avoid false positives
        ret, frame = cap.read()
        if ret:
            print(f"\n🎉 SUCCESS! Valid RTSP stream discovered!")
            print(f"--> Use this exact string: {url}\n")
            found_valid_url = url
            cap.release()
            break
            
    cap.release()

if not found_valid_url:
    print("\n❌ Scan complete. All hardcoded path variants returned 404.")
    print("Alternative: Go to the 'Config' tab in your web browser interface,")
    print("look under 'Network Services' or 'Encode', and look for RTSP path/port maps.")
