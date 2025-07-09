# Autonomous Tello Drone Landing using Visual Servoing and Search Strategy

This project simulates and implements autonomous landing of a DJI Tello drone using **blue marker detection** via OpenCV and a **square spiral search strategy** when the marker is lost. The early simulation was developed on gazebo and  it was developed as a prototype using `djitellopy` and is designed for testing in real environments or integration into Gazebo simulations with a vision-based drone model.

##  Features

-  Blue marker detection using HSV color thresholding.
-  Centering and descent based on marker position and size.
-  PID-like control for precise landing.
-  Square spiral search when marker is lost.
-  Visual display with error overlays, marker bounding boxes, and state info.
-  Safe landing logic on error or low battery.


##  Getting Started for prototype implementation

### Install Python & Dependencies

Ensure Python 3.7+ is installed. Then:

```bash
pip install -r requirements.txt

python main.py
