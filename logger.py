import pyautogui
import time
import sys

print("Log started. Press Ctrl+C to stop.")

try:
    while True:
        # Get the current mouse coordinates
        x, y = pyautogui.position()
        
        # Format the output with a timestamp
        current_time = time.strftime("%H:%M:%S")
        log_entry = f"[{current_time}] Mouse Location: X: {x} Y: {y}"
        
        # Print to the console
        print(log_entry)
        
        # Wait for 5 seconds before the next check
        time.sleep(5)

except KeyboardInterrupt:
    print("\nLogging stopped by user.")
    sys.exit()
