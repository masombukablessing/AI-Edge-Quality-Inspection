# AI-Edge-Quality-Inspection
Real-time defect detection using Edge AI for smart manufacturing

AI-Based Quality Inspection System Using Edge AI 🏭🤖

Hi there! 👋 I’m Janani, and this project is an AI-based Quality Inspection System that automates the process of detecting good and defective products in industries using Edge AI.

🌟 Overview

Manual quality inspection can be slow, error-prone, and inconsistent, especially in fast-paced production environments. This project solves that problem by automatically analyzing product images and identifying defects with high accuracy.

The system uses a Convolutional Neural Network (CNN) to classify images into good or defective, making industrial inspection faster, more reliable, and scalable. The results are saved in a structured format for further analysis and reporting.

This solution is ideal for factories, manufacturing units, and quality assurance teams who want to streamline inspection processes and reduce human errors.

✨ Features

📸 Image-Based Inspection – Automatically detects defects in products using captured images.
🤖 Edge AI Deployment – Can run on local devices for real-time inspection without heavy cloud dependency.
🗂️ Organized Results – Saves classified images in good/ and defective/ folders for easy review.
📊 Performance Tracking – Optionally logs predictions in CSV files and generates accuracy/loss plots.
⚡ Fast & Scalable – Handles large batches of images for industrial-scale inspection.

🛠️ Tech Stack

1.Programming Language: Python
2.Deep Learning Framework: TensorFlow / Keras
3.Data Handling: ImageDataGenerator, CSV logging
4.File Management: Organized folder structure (dataset/, models/, results/)
5.Optional Visualization: Accuracy/Loss graphs, Confusion Matrix

📂 Folder Structure

AI_Inspection_System/
├── dataset/
│   ├── good_images/
│   └── defective_images/
├── models/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── saved_model/
├── results/
│   └── predictions/
│       ├── good/
│       └── defective/
└── README.md

✅ Usage

1.Train the model using train.py with your dataset.
2.Run predictions using evaluate.py or a batch prediction script.
3.Check results/predictions/good and results/predictions/defective for classified images.
4.Optionally, export a CSV file with all predictions for reporting.

🌐 Applications

1.Industrial product inspection.
2.Quality assurance automation.
3.Manufacturing workflow optimization.
4.Edge AI applications in factories.

🎯 Outcome

By automating the inspection process, this system reduces human error, increases inspection speed, and provides structured data for quality analysis. Perfect for showcasing AI-powered industrial solutions in your portfolio or GitHub repo.
