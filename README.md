# Binarization of historical documents

## Brief Description

A Python application which uses Sauvola’s method for binarization of historical documents. Implementing and tuning the algorithm on integral images produced better results in minimal time.


## Problem

Scanned documents are being used in majority of the employment sectors. The fact that they are not always clear and more often than not, result in low readability is a major let-down. Moreover, they do not provide search functionality. This makes it harder to find a topic of interest in a document. Hence, there arises a need for machine assistance to cater to this problem.

Binarization of images of ancient documents is the process of removing unrelated artefacts and background noise from a particular image. Image binarization is essential for image analysis as it significantly improve the overall readability and information segmentation of the original characters in the image. Determining proper binarization techniques is a key factor in achieving promising results from document image analysis.


## Solution

My solution focuses on binarization. I convert the image to greyscale, and then instead of doing global binarization as in a traditional method, I develop on Sauvola's algorithm which finds the appropriate threshold and binarizes. For information segmentation, I tune the algorithm in such a way that it distinguishes and detects different characters and then recognizes them automatically. This project caters to all who use scanned documents for any purpose i.e study, work or research. The main dependency of this project is OpenCV 4.0.