# Bone Fracture Detection Using Image Processing (MATLAB & Python) - Mini Project at Undergrad and personal project for developing python skills

This repository contains a student project on **automated bone fracture indication from X-ray images** using classical image-processing techniques in both **MATLAB** and **Python (OpenCV)**.

Disclaimer: This is a **research/educational project only** and **not a clinical tool**. It is not intended for medical diagnosis.

Project Idea

Starting from a basic **Hough Transform–based line detection** mini-project in MATLAB, this project was extended and translated into Python to:
- Preprocess X-ray images (contrast enhancement, denoising)
- Extract edges using Canny edge detection
- Detect cortical bone line segments using the Hough Transform
- Estimate the **main bone axis**
- Highlight **short, off-angle line segments** as *possible fracture candidates*
