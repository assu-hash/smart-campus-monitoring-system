import cv2
import numpy as np
import datetime
import os
import time
from threading import Thread


class SmartCampusSurveillance:
    def __init__(self, camera_id=0, sensitivity=20, record_seconds_after_motion=10):
        """
        Initialize the Smart Campus Surveillance System

        Args:
            camera_id: Camera device ID (default: 0)
            sensitivity: Motion detection sensitivity (default: 20)
            record_seconds_after_motion: Continue recording for seconds after motion stops (default: 10)
        """
        self.camera_id = camera_id
        self.sensitivity = sensitivity
        self.record_seconds_after_motion = record_seconds_after_motion
        self.is_recording = False
        self.motion_detected = False
        self.last_motion_time = 0

        # Create output directory if it doesn't exist
        self.output_dir = "surveillance_recordings"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Video capture and writer objects
        self.cap = None
        self.out = None

        # Motion detection parameters
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)
        self.min_contour_area = 500

    def start(self):
        """Start the surveillance system"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print("Smart Campus Surveillance System Started")
        print("Press 'q' to quit")

        # Start the processing loop
        self.process_video_feed()

        # Clean up
        self.cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()

    def detect_motion(self, frame):
        """
        Detect motion in frame

        Args:
            frame: Current video frame

        Returns:
            motion_detected: Boolean indicating if motion was detected
            frame_with_boxes: Frame with bounding boxes around motion areas
        """
        # Create a copy of the frame
        frame_with_boxes = frame.copy()

        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Threshold the mask
        _, thresh = cv2.threshold(fg_mask, self.sensitivity, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to reduce noise
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize motion detection flag
        motion_detected = False

        for contour in contours:
            # Filter out small contours
            if cv2.contourArea(contour) < self.min_contour_area:
                continue

            # Motion detected
            motion_detected = True

            # Draw bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return motion_detected, frame_with_boxes

    def start_recording(self):
        """Start recording video"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"motion_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                   (self.frame_width, self.frame_height))
        self.is_recording = True
        print(f"Recording started: {output_path}")

    def stop_recording(self):
        """Stop recording video"""
        if self.out is not None:
            self.out.release()
            self.out = None
            self.is_recording = False
            print("Recording stopped")

    def add_timestamp(self, frame):
        """Add timestamp to frame"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        return frame

    def process_video_feed(self):
        """Process video feed for motion detection and recording"""
        # Skip the first few frames to allow background subtractor to initialize
        for _ in range(50):
            ret, _ = self.cap.read()
            if not ret:
                return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect motion
            motion_detected, frame_with_boxes = self.detect_motion(frame)

            # Add timestamp
            frame_with_boxes = self.add_timestamp(frame_with_boxes)

            # Handle motion detection state
            if motion_detected:
                self.motion_detected = True
                self.last_motion_time = time.time()

                # Start recording if not already recording
                if not self.is_recording:
                    self.start_recording()

                # Add "Motion Detected" indicator
                cv2.putText(frame_with_boxes, "Motion Detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # Continue recording for a few seconds after motion stops
                if self.motion_detected and time.time() - self.last_motion_time > self.record_seconds_after_motion:
                    self.motion_detected = False
                    if self.is_recording:
                        self.stop_recording()

            # Save frame if recording
            if self.is_recording:
                self.out.write(frame)

                # Add recording indicator
                cv2.putText(frame_with_boxes, "Recording", (self.frame_width - 140, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Smart Campus Surveillance', frame_with_boxes)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


class SmartCampusGuidance:
    def __init__(self, surveillance_system):
        """
        Initialize the Smart Campus Guidance System

        Args:
            surveillance_system: Instance of SmartCampusSurveillance
        """
        self.surveillance = surveillance_system
        self.crowd_threshold = 5  # Number of motion areas to indicate crowding

    def analyze_crowd_density(self, frame):
        """
        Analyze crowd density in frame

        Args:
            frame: Current video frame

        Returns:
            crowd_density: Estimated crowd density (low, medium, high)
            count: Number of detected motion areas
        """
        _, frame_with_boxes = self.surveillance.detect_motion(frame)

        # Count number of bounding boxes (people)
        contours, _ = cv2.findContours(
            self.surveillance.background_subtractor.apply(frame),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        count = 0
        for contour in contours:
            if cv2.contourArea(contour) >= self.surveillance.min_contour_area:
                count += 1

        # Determine crowd density
        if count <= 2:
            return "Low", count
        elif count <= 5:
            return "Medium", count
        else:
            return "High", count

    def suggest_route(self, current_location, destination, crowd_data):
        """
        Suggest optimal route based on crowd density

        Args:
            current_location: Starting location
            destination: Target location
            crowd_data: Dictionary with crowd density data for different areas

        Returns:
            route: Suggested route to destination
        """
        # This is a simplified implementation
        # In a real system, this would use pathfinding algorithms
        # and a campus map with real-time crowd data

        # Example implementation
        print(f"Finding route from {current_location} to {destination}")
        print("Crowd densities:")
        for location, density in crowd_data.items():
            print(f"- {location}: {density}")

        # Sample logic - avoid high density areas
        high_density_areas = [loc for loc, density in crowd_data.items() if density == "High"]

        if high_density_areas:
            return f"Route from {current_location} to {destination}: Avoiding {', '.join(high_density_areas)}"
        else:
            return f"Direct route from {current_location} to {destination} recommended (no crowded areas)"


class CampusSecurityAlert:
    def __init__(self, surveillance_system):
        """
        Initialize the Campus Security Alert System

        Args:
            surveillance_system: Instance of SmartCampusSurveillance
        """
        self.surveillance = surveillance_system
        self.alert_threshold = 10  # Seconds of continuous motion to trigger alert
        self.restricted_areas = []  # List of coordinates defining restricted areas
        self.active_alerts = []

    def define_restricted_area(self, x, y, width, height, name):
        """
        Define a restricted area

        Args:
            x, y: Top-left coordinates
            width, height: Dimensions
            name: Identifier for the area
        """
        self.restricted_areas.append({
            'coords': (x, y, width, height),
            'name': name
        })
        print(f"Restricted area defined: {name}")

    def check_for_intrusion(self, frame):
        """
        Check if motion is detected in restricted areas

        Args:
            frame: Current video frame

        Returns:
            intrusions: List of detected intrusions
        """
        intrusions = []

        # Get motion contours
        fg_mask = self.surveillance.background_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, self.surveillance.sensitivity, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check each contour against restricted areas
        for contour in contours:
            if cv2.contourArea(contour) < self.surveillance.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            motion_center = (x + w // 2, y + h // 2)

            for area in self.restricted_areas:
                ax, ay, aw, ah = area['coords']

                # Check if motion center is in restricted area
                if (ax <= motion_center[0] <= ax + aw) and (ay <= motion_center[1] <= ay + ah):
                    intrusions.append({
                        'area': area['name'],
                        'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'location': motion_center
                    })

        return intrusions

    def send_alert(self, intrusion):
        """
        Send security alert (placeholder function)

        Args:
            intrusion: Intrusion details
        """
        # In a real system, this would send email, SMS, or notify security personnel
        alert_id = len(self.active_alerts) + 1
        self.active_alerts.append({
            'id': alert_id,
            'details': intrusion,
            'status': 'ACTIVE'
        })

        print(f"⚠ SECURITY ALERT ⚠")
        print(f"Motion detected in restricted area: {intrusion['area']}")
        print(f"Time: {intrusion['time']}")
        print(f"Alert ID: {alert_id}")

        return alert_id

    def resolve_alert(self, alert_id):
        """
        Mark alert as resolved

        Args:
            alert_id: ID of alert to resolve
        """
        for alert in self.active_alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'RESOLVED'
                print(f"Alert {alert_id} marked as resolved")
                return True

        print(f"Alert {alert_id} not found")
        return False


def main():
    # Create and start surveillance system
    surveillance = SmartCampusSurveillance(camera_id=0, sensitivity=25)

    # Initialize guidance system
    guidance = SmartCampusGuidance(surveillance)

    # Initialize security alerts
    security = CampusSecurityAlert(surveillance)

    # Define restricted areas (example)
    security.define_restricted_area(50, 50, 200, 200, "Server Room")
    security.define_restricted_area(400, 300, 150, 100, "Admin Office")

    # Start surveillance in a separate thread
    surveillance_thread = Thread(target=surveillance.start)
    surveillance_thread.daemon = True
    surveillance_thread.start()

    # Main application loop
    try:
        print("System running. Press Ctrl+C to exit.")

        # Example crowd data (in a real system, this would be dynamically updated)
        crowd_data = {
            "Library": "High",
            "Cafeteria": "Medium",
            "Main Hall": "Low",
            "Engineering Building": "Medium"
        }

        # Example route guidance
        route = guidance.suggest_route("Main Entrance", "Library", crowd_data)
        print("\nGuidance Example:")
        print(route)

        # Keep the main thread running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nExiting system...")

    # Wait for surveillance thread to finish
    surveillance_thread.join(timeout=1.0)
    print("System shutdown complete")


if __name__ == "__main__":
    main()