# QUANTUM-SIMULATOR_KEY-GENERATION

This project explores entropy generation using a laptop webcam as the randomness source. Each frame, the system selects a random geometric shape (circle, rectangle, ellipse) as a measurement region, extracts pixel intensity values, and computes their Shannon entropy. These entropy values capture the unpredictable variations in the video feed caused by lighting, motion, and sensor noise.

To simulate quantum measurement, the sampled data is combined with additional system randomness and collapsed into a 256-bit cryptographic key using SHA-256. The process draws inspiration from the idea of wavefunction collapse, where measurement turns uncertainty into a definite outcome.

A live entropy graph shows how randomness fluctuates over time. Simple experiments, such as covering the webcam or shining a flashlight, visibly shift the entropy values, demonstrating the sensitivity of the setup to environmental changes.

This is an early prototype, developed in the first semester of undergraduate study, with planned updates including data logging, statistical randomness testing, and extended analysis.
(The source code is yet to be published and will not be published until a more polished version is created)
