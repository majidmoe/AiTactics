Football Tactical Analysis System

An AI-powered system for analyzing football match footage to provide coaches with tactical insights and actionable recommendations. This tool helps coaches develop effective strategies against specific opponents by identifying patterns, vulnerabilities, and opportunities.

Features

Player Tracking: Automatically detect and track players throughout the match

Formation Analysis: Identify team formations (4-4-2, 4-3-3, etc.)

Defensive Analysis: Detect vulnerabilities like high defensive lines, poor compactness, and slow transitions

Offensive Analysis: Identify attacking patterns and strategies

Tactical Recommendations: Generate specific, actionable recommendations to exploit opponent weaknesses

Visual Explanations: Create tactical diagrams that illustrate recommended strategies

Export Options: Save analysis as PDF reports or CSV data

Installation

Prerequisites

Python 3.10 or later

Git (for cloning the repository)

Step 1: Clone the Repository

git clone https://github.com/your-username/AiTactics.git
cd AiTactics


Step 2: Create a Virtual Environment

# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate


Step 3: Install Dependencies

pip install -r requirements.txt


If requirements.txt is not available, install the following packages:

pip install opencv-python==4.11.0 opencv-contrib-python==4.11.0
pip install torch==2.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.26.4 pandas==2.0.3 streamlit==1.24.0
pip install matplotlib==3.7.2 seaborn==0.13.2 fpdf==1.7.2
pip install ultralytics==8.3.92 scipy==1.15.2 plotly==5.15.0
pip install scikit-learn


Usage

Running the Application

streamlit run main/main.py


Analyzing a Match

Upload Match Video: Use the file uploader in the sidebar to upload your match footage

Configure Analysis Parameters:

Enter team names

Select team colors

Choose which team to focus on

Select analysis aspects (defensive structure, offensive patterns, etc.)

Start Analysis: Click the "Start Analysis" button

Review Results: Navigate through the tabs to view:

Formation analysis

Defensive vulnerabilities

Offensive patterns

Tactical recommendations

Exporting Analysis

Use the export options at the bottom of the Recommendations tab to save your analysis as:

PDF Report

CSV Data

PowerPoint Presentation

Project Structure

AiTactics/
├── main/                  # Main source code directory
│   ├── main.py            # Streamlit application
│   
├── model/                 # Pre-trained models
│   └── best (2).pt            # YOLOv8 model for player detection
├── venv/                  # Virtual environment (not tracked in git)
├── vids/                  # Example videos to use
├── requirements.txt       # Project dependencies
└── README.md              # This documentation


Common Issues and Troubleshooting

OpenCV Legacy Module Error

If you encounter an error about cv2.legacy not existing:

AttributeError: module 'cv2' has no attribute 'legacy'


Solution: Install opencv-contrib-python which includes the tracking modules:

pip install opencv-contrib-python


If you still encounter issues, modify the create_tracker method in player_tracker.py to use direct tracker creation methods.

Progress Bar Error

If you see an error like:

StreamlitAPIException: Progress Value has invalid value [0.0, 1.0]: 1.001020


Solution: Modify line 152 in main.py to ensure progress doesn't exceed 1.0:

progress_bar.progress(min(frame_count / total_frames, 1.0))


Float Object Not Subscriptable

If you encounter:

TypeError: 'float' object is not subscriptable


Solution: Check the _calculate_defensive_line_height method in tactical_analyzer.py. Ensure you're not trying to index a float value.

Contributing

We welcome contributions to improve the Football Tactical Analysis System!

How to Contribute

Fork the repository

Create a feature branch: git checkout -b feature/amazing-feature

Make your changes

Run tests to ensure your changes don't break existing functionality

Commit your changes: git commit -m 'Add amazing feature'

Push to your branch: git push origin feature/amazing-feature

Open a Pull Request

Development Guidelines

Follow PEP 8 style guidelines for Python code

Add comments to explain complex logic

Update documentation when adding new features

Include docstrings for new functions and classes

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

YOLO by Ultralytics for player detection

OpenCV for video processing and tracking

Streamlit for the web interface

All contributors who have helped improve this system

Contact

For questions or support, please open an issue on the GitHub repository or contact the project maintainer.