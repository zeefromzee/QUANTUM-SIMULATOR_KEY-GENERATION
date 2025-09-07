# Quantum-Inspired Key Generator

## Overview

The **Quantum-Inspired Key Generator** is a Python project that produces cryptographic keys by extracting entropy from a live webcam feed.
Instead of relying on pseudo-random number generators, this program introduces unpredictability by combining video noise with randomly chosen geometric shapes, then processing the data using SHA-256 hashing.

This is an **educational project** that demonstrates how randomness can be harvested from physical sources for cryptographic purposes.

---

## Features

* Uses live webcam feed as a source of entropy.
* Randomly selects a "measurement basis" (rectangle, circle, or ellipse).
* Extracts pixel values from the chosen region in real time.
* Processes entropy with SHA-256 to generate secure keys.
* Displays both the selected basis and partial key on the video feed.
* Runs on any system with Python, OpenCV, and NumPy.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/quantum-key-generator.git
cd quantum-key-generator
pip install opencv-python numpy
```

---

## Usage

Run the script:

```bash
python3 projj.py
```

* A new window will display the webcam feed.
* Random shapes will appear on the video (rectangle, circle, ellipse).
* A key is generated from the pixel data inside the shape and shown on screen.
* Press **ESC** to exit.

Example (on video overlay):

```
Quantum Key: a7d92f1c5b13d8e2
Measurement Basis: circle
```

---

## Methodology

1. Webcam frames are captured and converted to grayscale.
2. A random geometric shape is drawn at a random location.
3. Pixel data inside the shape is extracted as entropy.
4. Additional system randomness is added.
5. The combined data is hashed using **SHA-256**, producing a cryptographic key.

### Simplified Workflow

```
Webcam Feed → Random Shape → Pixel Extraction → Salt + SHA-256 → Key
```

---

## Future Improvements

* Export generated keys to a file.
* Allow user control over refresh rate and shape selection.
* Support for multiple webcams or other entropy sources.
* Add randomness quality testing (e.g., NIST tests).

---

## Disclaimer

This project does not use real quantum hardware.
It is intended for educational and experimental purposes only and should not be relied on in production cryptographic systems.

---

## License

This project is licensed under the MIT License.
