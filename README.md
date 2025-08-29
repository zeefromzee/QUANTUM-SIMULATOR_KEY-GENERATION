# Quantum-Inspired Key Generator

Overview

A quantum-inspired entropy generator using a laptop webcam as the randomness source. Random geometric masks extract pixel regions, entropy is computed using Shannon’s formula, and the results are collapsed into 256-bit cryptographic keys with SHA-256. A live graph shows entropy fluctuations influenced by environmental changes such as light and occlusion.

Project Description

This project explores entropy generation using a webcam as the entropy source. Each frame, the system selects a random geometric shape (circle, rectangle, or ellipse) as a measurement region, extracts pixel intensity values, and computes their Shannon entropy. These entropy values capture unpredictable variations in the video feed caused by lighting, motion, and sensor noise.

To simulate quantum measurement, the sampled data is combined with additional system randomness and collapsed into a 256-bit cryptographic key using SHA-256. The process draws inspiration from the concept of wavefunction collapse, where measurement turns uncertainty into a definite outcome.

A live entropy graph illustrates how randomness fluctuates over time. Simple experiments, such as covering the webcam or shining a flashlight, visibly shift the entropy values, demonstrating the sensitivity of the setup to environmental conditions.

This is an early prototype developed in the first semester of undergraduate study, with planned updates to expand functionality and improve reliability.

Future Work

Add data logging of entropy values and generated keys

Implement statistical randomness tests (NIST and Dieharder)

Enhance visualization and graph styling for clarity

Explore cryptographic use cases for generated keys

Package the project for reproducible builds and wider testing

Status

Early prototype (v1.0). Source code is under development and will be released once a more stable version is ready.
