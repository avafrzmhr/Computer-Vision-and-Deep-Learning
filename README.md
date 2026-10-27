# Computer Vision, 3D Geometry & Deep Learning Suite

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-lightblue)

## 📌 Overview
This repository contains a comprehensive collection of **Computer Vision**, **Deep Learning**, **3D Projective Geometry**, and **Digital Image Processing** implementations developed as part of the *Fundamentals of Computer Vision* coursework at **Babol Noshirvani University of Technology (BNUT)**. 

The implementations focus on building core mathematical and algorithmic foundations from scratch alongside deep neural network architectures using **TensorFlow/Keras**, **OpenCV**, **NumPy**, **SciPy**, and **Scikit-learn**.

---

## 📂 Repository Structure & Key Implementations

### 1. Deep Learning for Visual Recognition (`01_CNN_Classification_CIFAR10/`)
* **`cifar10_grayscale_cnn.py`**: 
  * Designed and trained a custom 3-block **Convolutional Neural Network (CNN)** on standardized grayscale CIFAR-10 images.
  * **Preprocessing & Augmentation:** Implemented RGB-to-luminance grayscale conversion, per-image standardization, horizontal flipping, and intensity scaling via `ImageDataGenerator`.
  * **Architecture & Regularization:** Incorporated $3 \times 3$ convolutions, **Batch Normalization**, $L_2$ weight decay ($\lambda = 0.001$), Max/Average Pooling, and Dense classification layers trained with the Adam optimizer.
  * **Evaluation:** Evaluated generalization performance across 100 epochs using test accuracy and weighted **F1-score**.

### 2. 3D Computer Vision & Projective Geometry (`02_3D_Vision_and_Geometry/`)
* **`camera_calibration_svd.py` (3D-to-2D Camera Calibration):**
  * Estimated the $3 \times 4$ **Camera Projection Matrix** from matched 2D image coordinates and 3D world coordinates by formulating a homogeneous linear system ($A\mathbf{p} = \mathbf{0}$) solved via **Singular Value Decomposition (SVD)**.
  * Extracted intrinsic camera parameters (including **Focal Length** via matrix inversion) and evaluated 3D-to-2D projection accuracy using **Mean Squared Error (MSE)**.
* **`perspective_homography.py` (Planar Homography & Perspective Warping):**
  * Computed $3 \times 3$ perspective transformation matrices from four-point correspondences to rectify angled views into fronto-parallel projections.

### 3. Frequency Domain Analysis & Image Restoration (`03_Frequency_Domain_and_Restoration/`)
* **`fft_phase_magnitude_swap.py` (2D Fast Fourier Transform Reconstruction):**
  * Decomposed images into **Magnitude** and **Phase** spectra using 2D FFT (`np.fft.fft2`).
  * Demonstrated the structural dominance of phase information in human visual perception by swapping phase and magnitude spectra between distinct images prior to Inverse FFT (IFFT) reconstruction.
* **`fft_frequency_filtering.py` (Frequency-Domain Filtering):**
  * Applied spectral masking in the centered Fourier domain (`fftshift`) to attenuate specific frequency bands and reconstruct filtered spatial images.
* **`adaptive_local_denoising.py` (Adaptive Local Noise Reduction):**
  * Implemented an adaptive spatial denoising filter from scratch based on local mean ($\mu_L$) and local variance ($\sigma_L^2$) estimation to preserve edges while suppressing additive Gaussian noise:
    $$\hat{f}(x,y) = g(x,y) - \frac{\sigma_\eta^2}{\sigma_L^2} \left(g(x,y) - \mu_L\right)$$
* **`multi_observation_denoising.py` (Temporal/Ensemble Noise Averaging):**
  * Analyzed the asymptotic reduction of Mean Squared Reconstruction Error across $k \in \{1, 5, 10, 15, 20\}$ noisy image observations.

### 4. Industrial Inspection, Morphology & Medical Enhancement (`04_Feature_Detection_and_Morphology/`)
* **`gear_boundary_morphology.py` & `gear_hough_circle_detection.py`:**
  * Automated mechanical gear inspection using binary thresholding, morphological opening/closing, **Canny Edge Detection**, external contour extraction, and the **Hough Circle Transform** (`cv2.HoughCircles`) to localize circular gear features and radii.
* **`xray_highpass_enhancement.py` (Medical X-Ray Sharpening):**
  * Enhanced fine anatomical structures in medical X-ray imagery using **Gaussian High-Pass Filtering** and unsharp masking (weighted spatial combination).

### 5. Image Steganography & Contrast Enhancement (`05_Steganography_and_Enhancement/`)
* **`bitplane_steganography_psnr.py` (8-Layer Bit-Plane Slicing & Steganography):**
  * Embedded secret grayscale images across individual bit-planes (LSB to MSB, bits $0$ to $7$) of a cover image using bitwise masking operations.
  * Quantitatively analyzed the trade-off between payload visibility and image fidelity across all 8 bit-planes using **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)**.
* **`histogram_equalization.py`:**
  * Implemented global histogram equalization and ratio-based illumination normalization ($r(x,y) = \frac{f(x,y)}{g(x,y) + \epsilon}$).

---

## 🛠️ Tech Stack & Libraries
* **Deep Learning:** `TensorFlow`, `Keras`
* **Computer Vision & Image Processing:** `OpenCV (cv2)`, `Pillow (PIL)`, `scikit-image`
* **Scientific Computing & Linear Algebra:** `NumPy`, `SciPy`, `scikit-learn`
* **Visualization:** `Matplotlib`

---
