import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
from fpdf import FPDF
from ultralytics import YOLO
from scipy.spatial import distance, ConvexHull
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN, KMeans
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import warnings
import requests
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any, Union

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants for field dimensions (standard soccer field in meters)
FIELD_WIDTH = 105  # meters
FIELD_HEIGHT = 68  # meters

# Session state keys
KEY_ANALYSIS_COMPLETE = "analysis_complete"
KEY_PLAYER_STATS = "player_stats"
KEY_TEAM_STATS = "team_stats"
KEY_EVENTS = "events"
KEY_VIDEO_INFO = "video_info"  # Add this key for storing video information

@dataclass
class PassEvent:
    time: float
    from_player: int
    to_player: int
    team: str
    from_position: Tuple[int, int]
    to_position: Tuple[int, int]
    completed: bool = True
    length: float = 0.0
    zone_from: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    zone_to: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    direction: str = "forward"  # forward, backward, lateral
    pass_type: str = "ground"   # ground, lofted, through
    progressive: bool = False   # Moves ball significantly forward
    danger_zone: bool = False   # Pass to final third or penalty area
    xA: float = 0.0            # Expected assists value
    breaking_lines: bool = False  # Pass that breaks defensive lines
    switch_play: bool = False     # Long cross-field pass

@dataclass
class ShotEvent:
    time: float
    player: int
    team: str
    position: Tuple[int, int]
    target_goal: str
    on_target: bool = False
    goal: bool = False
    expected_goal: float = 0.0
    distance: float = 0.0
    angle: float = 0.0
    shot_type: str = "normal"  # normal, volley, header, free_kick, penalty
    scenario: str = "open_play"  # open_play, set_piece, counter_attack
    pressure: float = 0.0  # A measure of defensive pressure during shot (0-1)
    zone: str = "central"  # central, left_side, right_side, box, outside_box
    
@dataclass
class Formation:
    shape: str  # e.g. "4-3-3"
    positions: List[Tuple[int, int]]
    confidence: float = 0.0
    timestamp: float = 0.0
    team: str = ""
    
    def to_dict(self):
        return {
            "shape": self.shape,
            "positions": self.positions,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "team": self.team
        }

@dataclass
class VideoInfo:
    total_frames: int = 0
    fps: float = 30.0
    frame_width: int = 1280
    frame_height: int = 720
    target_width: int = 1280
    target_height: int = 720
    processed_frames: int = 0
    duration: float = 0.0

class FootballAnalyzer:
    def __init__(self):
        # Initialize session state for persistence across reruns
        self.initialize_session_state()
        
        st.set_page_config(page_title="AI Football Analysis Platform", page_icon="⚽", layout="wide")
        st.title("⚽ Advanced Football Match Analysis Platform")
        
        # Initialize sidebar
        self.setup_sidebar()
        
        # Set up tabs for different analyses
        self.setup_tabs()
        
        # Initialize video information with defaults
        self.video_info = VideoInfo()
        
        # Initialize data structures
        self.initialize_data_structures()
        
        try:
            # Initialize tactical engine
            self.initialize_tactical_engine()
        except Exception as e:
            st.warning(f"Could not initialize tactical engine: {str(e)}. Some features may be limited.")
        
        # Initialize model if settings are available
        if self.check_model_settings_complete():
            try:
                self.initialize_model()
            except Exception as e:
                st.sidebar.error(f"Error initializing model: {str(e)}")
    
    def initialize_session_state(self):
        """Initialize session state variables for persistence across reruns"""
        if 'page' not in st.session_state:
            st.session_state.page = 'main'
            
        if KEY_ANALYSIS_COMPLETE not in st.session_state:
            st.session_state[KEY_ANALYSIS_COMPLETE] = False
            
        if KEY_PLAYER_STATS not in st.session_state:
            st.session_state[KEY_PLAYER_STATS] = None
            
        if KEY_TEAM_STATS not in st.session_state:
            st.session_state[KEY_TEAM_STATS] = None
            
        if KEY_EVENTS not in st.session_state:
            st.session_state[KEY_EVENTS] = None
            
        if KEY_VIDEO_INFO not in st.session_state:
            st.session_state[KEY_VIDEO_INFO] = VideoInfo()
            
        if 'processed_video_path' not in st.session_state:
            st.session_state.processed_video_path = None
            
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = None
    
    def setup_sidebar(self):
        """Set up the sidebar with input options"""
        st.sidebar.title("⚙️ Analysis Settings")
        
        # Team information
        self.team_home = st.sidebar.text_input("🏠 Home Team Name", "Home Team")
        self.team_away = st.sidebar.text_input("🚀 Away Team Name", "Away Team")
        
        self.home_color = st.sidebar.color_picker("🎽 Home Team Jersey Color", "#0000FF")
        self.away_color = st.sidebar.color_picker("🎽 Away Team Jersey Color", "#FF0000")
        
        # Style of play description
        st.sidebar.markdown("### Team Playing Styles")
        self.home_playstyle = st.sidebar.selectbox(
            "Home Team Style", 
            ["Possession-Based", "Counter-Attack", "High-Press", "Defensive", "Direct Play", "Custom"],
            index=0
        )
        if self.home_playstyle == "Custom":
            self.home_playstyle_custom = st.sidebar.text_area("Describe Home Team Style", "")
            
        self.away_playstyle = st.sidebar.selectbox(
            "Away Team Style", 
            ["Possession-Based", "Counter-Attack", "High-Press", "Defensive", "Direct Play", "Custom"],
            index=1
        )
        if self.away_playstyle == "Custom":
            self.away_playstyle_custom = st.sidebar.text_area("Describe Away Team Style", "")
        
        # Analysis settings
        self.confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 0.9, 0.3)
        self.tracking_memory = st.sidebar.slider("Player Tracking Memory (frames)", 10, 200, 50)
        self.frame_skip = st.sidebar.slider("Process every N frames", 1, 10, 3)
        
        # Advanced settings with expander
        with st.sidebar.expander("Advanced Settings"):
            self.use_gpu = st.checkbox("Use GPU (if available)", True)
            self.use_half_precision = st.checkbox("Use Half Precision (FP16)", True)
            self.batch_size = st.slider("Batch Size", 1, 16, 4)
            self.iou_threshold = st.slider("IOU Threshold", 0.1, 0.9, 0.5)
            self.analysis_resolution = st.select_slider(
                "Analysis Resolution",
                options=["Low (360p)", "Medium (720p)", "High (1080p)"],
                value="Medium (720p)"
            )
            
            # Formation detection settings
            self.formation_detection_method = st.selectbox(
                "Formation Detection Method",
                ["Basic", "Clustering", "Enhanced"],
                index=2
            )
            
            # Pass detection sensitivity
            self.pass_detection_sensitivity = st.slider(
                "Pass Detection Sensitivity", 
                1, 10, 5,
                help="Higher values detect more passes but may include false positives"
            )
            
            # Enhanced shot detection sensitivity
            self.shot_detection_sensitivity = st.slider(
                "Shot Detection Sensitivity", 
                1, 10, 5,
                help="Higher values detect more shots but may include false positives"
            )
            
            # Limit frames for processing
            self.max_frames_to_process = st.slider(
                "Max Frames to Process", 
                500, 10000, 5000,
                help="Limit the number of frames to process for faster analysis"
            )
        
        # Model options
        with st.sidebar.expander("Model Settings"):
            self.model_source = st.radio(
                "Model Source",
                ["Default YOLOv8", "Upload Custom Model"],
                index=0
            )
            
            if self.model_source == "Upload Custom Model":
                self.custom_model_file = st.file_uploader("Upload Custom YOLO Model", type=["pt", "pth"])
                self.model_description = st.text_input("Model Description (optional)", "")
            else:
                self.custom_model_file = None
        
        # Video upload
        self.video_path = st.sidebar.file_uploader("📂 Upload Match Video", type=["mp4", "avi", "mov"])
        
        # Analysis options
        self.enable_heatmap = st.sidebar.checkbox("Generate Heatmaps", True)
        self.enable_formation = st.sidebar.checkbox("Analyze Team Formations", True)
        self.enable_events = st.sidebar.checkbox("Detect Key Events", True)
        self.enable_report = st.sidebar.checkbox("Generate PDF Report", True)
        self.enable_tactical = st.sidebar.checkbox("Generate Tactical Analysis", True)
        
        # Gemini API integration
        with st.sidebar.expander("AI Integration"):
            self.enable_gemini = st.checkbox("Enable Gemini AI Insights", False)
            if self.enable_gemini:
                st.session_state.gemini_api_key = st.text_input(
                    "Gemini API Key", 
                    value=st.session_state.gemini_api_key if st.session_state.gemini_api_key else "",
                    type="password"
                )
                self.gemini_model = st.selectbox(
                    "Gemini Model",
                    ["gemini-pro", "gemini-1.5-pro"],
                    index=1
                )
        
        # Start analysis button
        self.start_analysis = st.sidebar.button("🚀 Start Analysis")
        
        # Reset analysis button
        if st.session_state[KEY_ANALYSIS_COMPLETE]:
            if st.sidebar.button("🔄 Reset Analysis"):
                for key in [KEY_ANALYSIS_COMPLETE, KEY_PLAYER_STATS, KEY_TEAM_STATS, KEY_EVENTS]:
                    st.session_state[key] = None
                st.session_state[KEY_VIDEO_INFO] = VideoInfo()
                st.session_state[KEY_ANALYSIS_COMPLETE] = False
                st.session_state.processed_video_path = None
                st.experimental_rerun()
    
    def check_model_settings_complete(self):
        """Check if model settings are complete and valid"""
        if self.model_source == "Upload Custom Model" and not self.custom_model_file:
            return False
        return True
    
    def initialize_model(self):
        """Initialize YOLO model with optimizations"""
        with st.spinner("Loading models..."):
            device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
            
            if self.model_source == "Upload Custom Model" and self.custom_model_file is not None:
                # Save uploaded model file to temp location
                model_bytes = self.custom_model_file.read()
                temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
                temp_model_file.write(model_bytes)
                model_path = temp_model_file.name
                temp_model_file.close()
                
                st.sidebar.success(f"✅ Custom model loaded: {self.custom_model_file.name}")
            else:
                # Use default model path or download from Ultralytics hub
                model_path = "yolov8x-pose.pt"
                if not os.path.exists(model_path):
                    model_path = "yolov8x.pt"  # Fallback to standard model
            
            try:
                # Load model with optimizations
                self.model = YOLO(model_path)
                
                # Apply optimizations
                self.model.to(device)
                if self.use_half_precision and device == "cuda":
                    self.model.half()  # FP16 precision for faster inference
                    
                # Set model parameters
                self.model.conf = self.confidence_threshold
                self.model.iou = self.iou_threshold
                
                st.sidebar.success(f"✅ Model loaded successfully on {device.upper()}")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.stop()
    
    def initialize_data_structures(self):
        """Initialize all data structures for tracking and analysis"""
        # Load video info from session state if available
        if st.session_state[KEY_VIDEO_INFO]:
            self.video_info = st.session_state[KEY_VIDEO_INFO]
        
        # Player tracking
        self.player_positions = defaultdict(lambda: deque(maxlen=self.tracking_memory))
        self.player_velocities = defaultdict(lambda: deque(maxlen=self.tracking_memory))
        self.player_team = {}  # Map player ID to team
        self.speed_data = defaultdict(list)
        self.distance_data = defaultdict(float)
        self.acceleration_data = defaultdict(list)
        self.ball_possession = defaultdict(int)
        self.ball_positions = deque(maxlen=self.tracking_memory)
        self.ball_velocities = deque(maxlen=self.tracking_memory)
        
        # Team analysis
        self.team_possession_frames = {self.team_home: 0, self.team_away: 0}
        self.team_positions = {self.team_home: [], self.team_away: []}
        self.team_formations = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        
        # Enhanced formation tracking
        self.formation_history = {self.team_home: [], self.team_away: []}
        self.formation_transitions = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        
        # Event detection
        self.events = []
        self.pass_data = []
        self.shot_data = []
        self.defensive_actions = defaultdict(list)
        self.pressing_data = defaultdict(list)
        
        # Enhanced pass analysis
        self.pass_success_rate = {self.team_home: 0, self.team_away: 0}
        self.pass_types = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.pass_directions = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.pass_networks = {self.team_home: defaultdict(lambda: defaultdict(int)), 
                             self.team_away: defaultdict(lambda: defaultdict(int))}
        self.progressive_passes = {self.team_home: 0, self.team_away: 0}
        self.danger_zone_passes = {self.team_home: 0, self.team_away: 0}
        self.pass_length_distribution = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.breaking_lines_passes = {self.team_home: 0, self.team_away: 0}
        self.switch_play_passes = {self.team_home: 0, self.team_away: 0}
        self.total_xA = {self.team_home: 0.0, self.team_away: 0.0}
        
        # Enhanced shot analysis
        self.shot_types = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.shot_scenarios = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.shot_zones = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.total_xG = {self.team_home: 0.0, self.team_away: 0.0}
        self.shots_under_pressure = {self.team_home: 0, self.team_away: 0}
        self.shot_success_rate = {self.team_home: 0.0, self.team_away: 0.0}
        self.shot_efficiency = {self.team_home: 0.0, self.team_away: 0.0}  # Goals vs xG
        
        # Pitch zones (divide pitch into a 6x9 grid)
        self.zone_possession = np.zeros((6, 9, 2))  # Last dimension: [home, away]
        self.zone_passes = np.zeros((6, 9, 2))
        self.zone_shots = np.zeros((6, 9, 2))
        self.zone_defensive_actions = np.zeros((6, 9, 2))
        self.zone_pressure = np.zeros((6, 9, 2))

        # Team strengths and weaknesses
        self.team_strengths = {self.team_home: {}, self.team_away: {}}
        self.team_weaknesses = {self.team_home: {}, self.team_away: {}}
        
        # Tactical analysis data
        self.pressing_intensity = {self.team_home: 0, self.team_away: 0}
        self.defensive_line_height = {self.team_home: 0, self.team_away: 0}
        self.pass_length_data = {self.team_home: [], self.team_away: []}
        self.buildup_patterns = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.tactical_suggestions = {self.team_home: [], self.team_away: []}
        
        # Individual player roles and performance
        self.player_roles = {}
        self.player_performance = {}
        
        # Analysis results
        self.analysis_results = {}
        
        # For tracking
        self.prev_positions = {}
        self.prev_frame_time = None
        
        # Gemini API integration
        self.gemini_insights = {self.team_home: [], self.team_away: []}
        
        # Initialize player_stats_df with an empty DataFrame to avoid errors
        self.player_stats_df = pd.DataFrame()
        
        # Initialize team_stats with default values
        self.team_stats = {
            self.team_home: {
                'Possession (%)': 0,
                'Distance (m)': 0,
                'Passes': 0,
                'Shots': 0
            },
            self.team_away: {
                'Possession (%)': 0,
                'Distance (m)': 0,
                'Passes': 0,
                'Shots': 0
            }
        }
    
    def setup_tabs(self):
        """Set up tabs for different analyses"""
        self.tab1, self.tab2, self.tab3, self.tab4, self.tab5, self.tab6, self.tab7 = st.tabs([
            "📹 Video Analysis", 
            "🔍 Player Stats", 
            "🌍 Spatial Analysis",
            "📊 Team Analysis",
            "💪 Strengths & Weaknesses",
            "🎯 Tactical Suggestions",
            "📝 Report"
        ])
    
    def initialize_tactical_engine(self):
        """Initialize the tactical analysis engine"""
        # Define tactical patterns for different play styles
        self.playstyle_patterns = {
            "Possession-Based": {
                "pass_length": "short",
                "pass_tempo": "high",
                "defensive_line": "high",
                "pressing_intensity": "medium",
                "width": "wide",
                "counter_attack_speed": "low",
                "key_zones": [(2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]
            },
            "Counter-Attack": {
                "pass_length": "long",
                "pass_tempo": "low",
                "defensive_line": "low",
                "pressing_intensity": "low",
                "width": "narrow",
                "counter_attack_speed": "high",
                "key_zones": [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4)]
            },
            "High-Press": {
                "pass_length": "short",
                "pass_tempo": "high",
                "defensive_line": "high",
                "pressing_intensity": "high",
                "width": "wide",
                "counter_attack_speed": "medium",
                "key_zones": [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5)]
            },
            "Defensive": {
                "pass_length": "mixed",
                "pass_tempo": "low",
                "defensive_line": "low",
                "pressing_intensity": "low",
                "width": "narrow",
                "counter_attack_speed": "medium",
                "key_zones": [(4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5)]
            },
            "Direct Play": {
                "pass_length": "long",
                "pass_tempo": "medium",
                "defensive_line": "medium",
                "pressing_intensity": "medium",
                "width": "wide",
                "counter_attack_speed": "high",
                "key_zones": [(2, 0), (2, 8), (3, 0), (3, 8)]
            }
        }
        
        # Counter-strategies for each play style
        self.counter_strategies = {
            "Possession-Based": [
                "Apply high pressing to disrupt build-up play",
                "Maintain compact defensive shape to limit passing options",
                "Quick transitions when ball is won to exploit space behind high defensive line",
                "Target pressing triggers (back passes, slow sideways passes)",
                "Overload central areas to force play wide"
            ],
            "Counter-Attack": [
                "Maintain possession to limit counter-attacking opportunities",
                "Position players to prevent long clearances and transitions",
                "Use pressing traps in opponent's half",
                "Maintain defensive awareness even in attacking phases",
                "Apply quick counter-pressing when possession is lost"
            ],
            "High-Press": [
                "Use direct passes to bypass the press",
                "Position tall players for long balls",
                "Quick ball movement with one-touch passing",
                "Use goalkeeper as additional passing option",
                "Create numerical advantage in build-up phase with dropping midfielders"
            ],
            "Defensive": [
                "Patient build-up to draw out defensive block",
                "Use width to stretch defensive structure",
                "Quick switches of play to create space",
                "Utilize creative players between defensive lines",
                "Use set pieces effectively as scoring opportunities"
            ],
            "Direct Play": [
                "Maintain strong aerial presence in defense",
                "Position for second balls after long passes",
                "Apply pressure on wide areas to prevent crosses",
                "Maintain compact defensive shape vertically",
                "Use technical players to retain possession when winning the ball"
            ],
            "Custom": [
                "Analyze opponent patterns throughout the match",
                "Focus on exploiting spaces when opposition changes formation",
                "Adjust pressing strategy based on opponent build-up patterns",
                "Target transitions during opponent's attacking phases",
                "Adapt formation to counter opponent's key playmakers"
            ]
        }
        
        # Tactical weakness indicators
        self.tactical_weaknesses = {
            "possession_loss_own_half": "Vulnerable to high pressing",
            "low_possession_percentage": "Difficulty maintaining control",
            "low_pass_completion": "Inconsistent build-up play",
            "high_goals_conceded_counter": "Vulnerable to counter-attacks",
            "low_defensive_duels_won": "Weak in defensive duels",
            "high_crosses_conceded": "Vulnerable in wide areas",
            "low_aerial_duels_won": "Weak in aerial situations",
            "high_shots_conceded_box": "Poor box defense",
            "low_shots_on_target": "Inefficient attacking",
            "low_pressing_success": "Ineffective pressing system",
            "low_forward_passes": "Lacks attacking progression",
            "high_lateral_passes": "Predictable sideways passing",
            "low_progressive_passes": "Difficulty progressing up the field",
            "poor_defensive_transitions": "Vulnerable during defensive transitions",
            "low_xG_per_shot": "Poor shot quality selection"
        }
        
        # Tactical strength indicators
        self.tactical_strengths = {
            "high_possession_percentage": "Strong ball retention",
            "high_pass_completion": "Effective build-up play",
            "high_passes_final_third": "Creative in attack",
            "high_crosses_completed": "Effective wide play",
            "high_pressing_success": "Effective pressing system",
            "high_defensive_duels_won": "Strong in defensive duels",
            "high_aerial_duels_won": "Strong in aerial situations",
            "low_shots_conceded_box": "Solid box defense",
            "high_shots_on_target": "Efficient attacking",
            "high_counter_attacks": "Effective on transitions",
            "high_forward_passes": "Progressive attacking play",
            "high_through_balls": "Creative penetrative passing",
            "high_progressive_passes": "Excellent at advancing the ball",
            "strong_defensive_transitions": "Quick recovery after losing possession",
            "high_xG_per_shot": "Creates high-quality chances"
        }
        
        # Initialize with default values for pressing intensity
        self.pressing_intensity = {self.team_home: 60, self.team_away: 60}
    
    def preprocess_video(self, video_file):
        """Preprocess video file and return video capture"""
        # Save uploaded file to temp location
        if video_file is not None:
            try:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(video_file.read())
                video_path = temp_file.name
                temp_file.close()
                
                # Get video properties
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Error: Could not open video file.")
                    return None, None
                    
                # Store video info in class and session state
                self.video_info.fps = cap.get(cv2.CAP_PROP_FPS)
                self.video_info.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.video_info.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_info.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_info.duration = self.video_info.total_frames / self.video_info.fps
                
                # Resize based on selected resolution
                if self.analysis_resolution == "Low (360p)":
                    self.video_info.target_height = 360
                elif self.analysis_resolution == "Medium (720p)":
                    self.video_info.target_height = 720
                else:
                    self.video_info.target_height = 1080
                    
                scale = self.video_info.target_height / self.video_info.frame_height
                self.video_info.target_width = int(self.video_info.frame_width * scale)
                
                # Save video info to session state
                st.session_state[KEY_VIDEO_INFO] = self.video_info
                
                return cap, video_path
            except Exception as e:
                st.error(f"Error processing video file: {str(e)}")
                return None, None
        return None, None
    
    def draw_field_overlay(self, frame):
        """Draw a semi-transparent soccer field overlay on the frame"""
        # Create a copy of the frame for overlay
        overlay = frame.copy()
        
        # Draw field outline
        cv2.rectangle(overlay, (0, 0), (self.video_info.target_width, self.video_info.target_height), (0, 128, 0), -1)
        
        # Draw center circle
        center_x, center_y = self.video_info.target_width // 2, self.video_info.target_height // 2
        radius = min(self.video_info.target_width, self.video_info.target_height) // 10
        cv2.circle(overlay, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Draw center line
        cv2.line(overlay, (center_x, 0), (center_x, self.video_info.target_height), (255, 255, 255), 2)
        
        # Draw penalty areas (simplified)
        penalty_width = self.video_info.target_width // 6
        penalty_height = self.video_info.target_height // 4
        
        # Left penalty area
        cv2.rectangle(overlay, (0, center_y - penalty_height // 2), 
                     (penalty_width, center_y + penalty_height // 2), (255, 255, 255), 2)
        
        # Right penalty area
        cv2.rectangle(overlay, (self.video_info.target_width - penalty_width, center_y - penalty_height // 2), 
                     (self.video_info.target_width, center_y + penalty_height // 2), (255, 255, 255), 2)
        
        # Draw 6-yard boxes
        six_yard_width = self.video_info.target_width // 12
        six_yard_height = self.video_info.target_height // 6
        
        # Left 6-yard box
        cv2.rectangle(overlay, (0, center_y - six_yard_height // 2), 
                     (six_yard_width, center_y + six_yard_height // 2), (255, 255, 255), 2)
        
        # Right 6-yard box
        cv2.rectangle(overlay, (self.video_info.target_width - six_yard_width, center_y - six_yard_height // 2), 
                     (self.video_info.target_width, center_y + six_yard_height // 2), (255, 255, 255), 2)
        
        # Draw goal lines
        goal_width = self.video_info.target_width // 40
        goal_height = self.video_info.target_height // 8
        
        # Left goal
        cv2.rectangle(overlay, (0, center_y - goal_height // 2), 
                     (-goal_width, center_y + goal_height // 2), (255, 255, 255), 2)
        
        # Right goal
        cv2.rectangle(overlay, (self.video_info.target_width, center_y - goal_height // 2), 
                     (self.video_info.target_width + goal_width, center_y + goal_height // 2), (255, 255, 255), 2)
        
        # Draw zone grid (6x9)
        zone_width = self.video_info.target_width / 9
        zone_height = self.video_info.target_height / 6
        
        # Draw vertical lines
        for i in range(1, 9):
            x = int(i * zone_width)
            cv2.line(overlay, (x, 0), (x, self.video_info.target_height), (255, 255, 255), 1)
        
        # Draw horizontal lines
        for i in range(1, 6):
            y = int(i * zone_height)
            cv2.line(overlay, (0, y), (self.video_info.target_width, y), (255, 255, 255), 1)
        
        # Blend the original frame with the overlay
        alpha = 0.2  # Transparency factor
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    def analyze_frame(self, frame, frame_idx):
        """Analyze a single frame for player and ball detection"""
        if frame is None:
            return frame
            
        try:
            # Resize frame for efficiency
            frame_resized = cv2.resize(frame, (self.video_info.target_width, self.video_info.target_height))
            
            current_time = frame_idx / self.video_info.fps
            
            # Run YOLO detection with batch processing for efficiency
            try:
                # Use batch processing if available
                if hasattr(self.model, 'predict') and self.batch_size > 1:
                    results = self.model.predict(frame_resized, verbose=False, conf=self.confidence_threshold, 
                                               iou=self.iou_threshold, augment=False)
                else:
                    results = self.model(frame_resized, verbose=False)
                    
                if not results or len(results) == 0:
                    # Handle empty results
                    return self.draw_field_overlay(frame_resized)
                    
            except Exception as e:
                st.error(f"Error in model inference: {str(e)}")
                return frame_resized
            
            # Process detection results
            detected_players = []
            ball_detected = False
            ball_position = None
            
            # Draw field overlay for better visualization
            frame_resized = self.draw_field_overlay(frame_resized)
            
            # Process each detection
            for i, det in enumerate(results[0].boxes):
                # Get bounding box
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                conf = det.conf[0].cpu().numpy()
                
                # Get class (0=player, 1=ball, etc.)
                cls_id = int(det.cls[0].cpu().numpy())
                
                # Calculate center position
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # If detection is a player
                if cls_id == 0:  # Player class
                    # Assign player ID using tracking
                    player_id = i
                    
                    # Determine team based on jersey color matching
                    # This is a simplified approach - in a real system, use a jersey color classifier
                    if player_id not in self.player_team:
                        # Simple left/right side heuristic for team assignment
                        if center_x < self.video_info.target_width / 2:
                            self.player_team[player_id] = self.team_home
                        else:
                            self.player_team[player_id] = self.team_away
                    
                    team = self.player_team[player_id]
                    team_color = self.home_color if team == self.team_home else self.away_color
                    
                    # Convert hex color to BGR
                    color = tuple(int(team_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
                    
                    # Calculate player speed and distance
                    if player_id in self.prev_positions:
                        # Calculate time delta
                        dt = 1.0 / self.video_info.fps * self.frame_skip
                        
                        prev_x, prev_y = self.prev_positions[player_id]
                        dx = center_x - prev_x
                        dy = center_y - prev_y
                        
                        # Calculate velocity vector
                        velocity = (dx/dt, dy/dt)  # pixels per second
                        self.player_velocities[player_id].append(velocity)
                        
                        # Calculate distance in pixels, then convert to meters using field ratio
                        distance_pixels = np.sqrt(dx**2 + dy**2)
                        
                        # Convert to real-world distance using field dimensions
                        pixel_to_meter = (FIELD_WIDTH / self.video_info.target_width + FIELD_HEIGHT / self.video_info.target_height) / 2
                        distance_meters = distance_pixels * pixel_to_meter
                        
                        # Calculate speed (m/s)
                        speed = distance_meters / dt
                        
                        # Calculate acceleration if we have previous speed data
                        if player_id in self.speed_data and len(self.speed_data[player_id]) > 0:
                            prev_speed = self.speed_data[player_id][-1]
                            acceleration = (speed - prev_speed) / dt
                            self.acceleration_data[player_id].append(acceleration)
                        
                        # Update data
                        self.distance_data[player_id] += distance_meters
                        self.speed_data[player_id].append(speed)
                        
                        # Track position in zone grid for zone analysis
                        zone_x = min(int(center_x / self.video_info.target_width * 9), 8)
                        zone_y = min(int(center_y / self.video_info.target_height * 6), 5)
                        team_idx = 0 if team == self.team_home else 1
                        self.zone_possession[zone_y, zone_x, team_idx] += 1
                    
                    # Store current position for next frame
                    self.prev_positions[player_id] = (center_x, center_y)
                    self.player_positions[player_id].append((int(center_x), int(center_y)))
                    
                    # Draw player bounding box and ID
                    cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame_resized, f"{player_id}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add to detected players
                    detected_players.append({
                        'id': player_id,
                        'team': team,
                        'position': (center_x, center_y),
                        'bbox': (x1, y1, x2, y2)
                    })
                    
                # If detection is a ball
                elif cls_id == 1:  # Ball class
                    ball_detected = True
                    ball_position = (int(center_x), int(center_y))
                    
                    # Calculate ball velocity if we have previous positions
                    if len(self.ball_positions) > 0:
                        prev_ball_x, prev_ball_y = self.ball_positions[-1]
                        dt = 1.0 / self.video_info.fps * self.frame_skip
                        
                        ball_dx = center_x - prev_ball_x
                        ball_dy = center_y - prev_ball_y
                        
                        # Store ball velocity vector
                        self.ball_velocities.append((ball_dx/dt, ball_dy/dt))
                    
                    self.ball_positions.append(ball_position)
                    
                    # Draw ball
                    cv2.circle(frame_resized, ball_position, 10, (0, 255, 255), -1)
                    
                    # Determine ball possession based on proximity
                    if detected_players:
                        nearest_player = min(detected_players, key=lambda p: 
                                            distance.euclidean(p['position'], ball_position))
                        self.ball_possession[nearest_player['id']] += 1
                        team = nearest_player['team']
                        self.team_possession_frames[team] += 1
                        
                        # Draw line from nearest player to ball
                        cv2.line(frame_resized, 
                                (int(nearest_player['position'][0]), int(nearest_player['position'][1])),
                                ball_position, (0, 255, 0), 2)
            
            # Detect and analyze team formations
            if self.enable_formation and len(detected_players) > 0:
                self.analyze_formations(detected_players, frame_resized, frame_idx)
            
            # Draw trajectories for players
            for player_id, positions in self.player_positions.items():
                if len(positions) > 1 and player_id in self.player_team:
                    team = self.player_team[player_id]
                    color = tuple(int(self.home_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0)) if team == self.team_home else \
                           tuple(int(self.away_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
                    
                    for i in range(1, len(positions)):
                        cv2.line(frame_resized, positions[i-1], positions[i], color, 1)
            
            # Detect key events
            if self.enable_events and ball_position:
                self.detect_events(frame_idx, detected_players, ball_position)
            
            # Display frame time
            cv2.putText(frame_resized, f"Time: {current_time:.2f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame_resized
        except Exception as e:
            st.error(f"Error in frame analysis: {str(e)}")
            traceback.print_exc()
            return frame
    
    def analyze_formations(self, detected_players, frame, frame_idx):
        """Analyze and visualize team formations with enhanced algorithms"""
        try:
            # Group players by team
            home_players = [p for p in detected_players if p['team'] == self.team_home]
            away_players = [p for p in detected_players if p['team'] == self.team_away]
            
            # Process home team formation
            if len(home_players) > 5:  # Need at least a few players to analyze formation
                self.team_positions[self.team_home].append([p['position'] for p in home_players])
                
                # Determine formation using the selected method
                if self.formation_detection_method == "Basic":
                    formation, confidence = self.calculate_formation_basic(home_players)
                elif self.formation_detection_method == "Clustering":
                    formation, confidence = self.calculate_formation_clustering(home_players)
                else:  # Enhanced method
                    formation, confidence, positions = self.calculate_formation_enhanced(home_players)
                    
                    # Store formation history with timestamp
                    formation_data = Formation(
                        shape=formation,
                        positions=positions,
                        confidence=confidence,
                        timestamp=frame_idx / self.video_info.fps,
                        team=self.team_home
                    )
                    self.formation_history[self.team_home].append(formation_data)
                    
                    # Track formation transitions
                    if len(self.formation_history[self.team_home]) > 1:
                        prev_formation = self.formation_history[self.team_home][-2].shape
                        current_formation = formation
                        if prev_formation != current_formation:
                            transition_key = f"{prev_formation}->{current_formation}"
                            self.formation_transitions[self.team_home][transition_key] += 1
                
                self.team_formations[self.team_home][formation] += 1
                
                # Draw formation lines on frame
                self.draw_formation_lines(frame, home_players, 
                                        tuple(int(self.home_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0)))
            
            # Process away team formation  
            if len(away_players) > 5:
                self.team_positions[self.team_away].append([p['position'] for p in away_players])
                
                # Determine formation using the selected method
                if self.formation_detection_method == "Basic":
                    formation, confidence = self.calculate_formation_basic(away_players)
                elif self.formation_detection_method == "Clustering":
                    formation, confidence = self.calculate_formation_clustering(away_players)
                else:  # Enhanced method
                    formation, confidence, positions = self.calculate_formation_enhanced(away_players)
                    
                    # Store formation history with timestamp
                    formation_data = Formation(
                        shape=formation,
                        positions=positions,
                        confidence=confidence,
                        timestamp=frame_idx / self.video_info.fps,
                        team=self.team_away
                    )
                    self.formation_history[self.team_away].append(formation_data)
                    
                    # Track formation transitions
                    if len(self.formation_history[self.team_away]) > 1:
                        prev_formation = self.formation_history[self.team_away][-2].shape
                        current_formation = formation
                        if prev_formation != current_formation:
                            transition_key = f"{prev_formation}->{current_formation}"
                            self.formation_transitions[self.team_away][transition_key] += 1
                
                self.team_formations[self.team_away][formation] += 1
                
                # Draw formation lines
                self.draw_formation_lines(frame, away_players, 
                                         tuple(int(self.away_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0)))
        except Exception as e:
            st.warning(f"Error in formation analysis: {str(e)}")
    
    def calculate_formation_basic(self, players):
        """Basic method to calculate team formation based on player positions"""
        try:
            # Sort players by y-coordinate (vertical position)
            sorted_players = sorted(players, key=lambda p: p['position'][1])
            
            # Count players in different thirds of the field (defense, midfield, attack)
            y_positions = [p['position'][1] for p in sorted_players]
            y_min, y_max = min(y_positions), max(y_positions)
            range_y = y_max - y_min if y_max > y_min else 1
            
            defenders = sum(1 for p in sorted_players if (p['position'][1] - y_min) / range_y < 0.33)
            midfielders = sum(1 for p in sorted_players if 0.33 <= (p['position'][1] - y_min) / range_y < 0.66)
            attackers = sum(1 for p in sorted_players if (p['position'][1] - y_min) / range_y >= 0.66)
            
            # Return formation as string (e.g., "4-3-3") and confidence value
            confidence = 0.7  # Basic method has a lower confidence by default
            return f"{defenders}-{midfielders}-{attackers}", confidence
        except Exception as e:
            print(f"Error in basic formation calculation: {e}")
            return "4-4-2", 0.5  # Default fallback
    
    def calculate_formation_clustering(self, players):
        """Calculate formation using clustering algorithms"""
        try:
            # Get player positions
            positions = np.array([list(p['position']) for p in players])
            
            if len(positions) < 5:  # Need at least 5 players for meaningful clustering
                return "Unknown", 0.4
            
            # Normalize positions to 0-1 range for better clustering
            x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
            y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
            
            range_x = x_max - x_min if x_max > x_min else 1
            range_y = y_max - y_min if y_max > y_min else 1
            
            normalized_positions = positions.copy()
            normalized_positions[:, 0] = (positions[:, 0] - x_min) / range_x
            normalized_positions[:, 1] = (positions[:, 1] - y_min) / range_y
            
            # Apply K-means clustering with k=3 for defense, midfield, attack
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(normalized_positions)
            
            # Count players in each cluster
            cluster_counts = np.bincount(clusters, minlength=3)
            
            # Sort clusters by y-position (vertical position)
            cluster_y_means = [np.mean(normalized_positions[clusters == i, 1]) for i in range(3)]
            sorted_indices = np.argsort(cluster_y_means)
            
            # Get player counts from back to front
            defenders = cluster_counts[sorted_indices[0]]
            midfielders = cluster_counts[sorted_indices[1]]
            attackers = cluster_counts[sorted_indices[2]]
            
            # Calculate confidence based on cluster separation
            confidence = min(1.0, 0.5 + kmeans.inertia_ / len(positions))
            
            # Return formation as string
            return f"{defenders}-{midfielders}-{attackers}", confidence
            
        except Exception as e:
            print(f"Clustering error: {str(e)}")
            # Fallback to basic method
            return self.calculate_formation_basic(players)
    
    def calculate_formation_enhanced(self, players):
        """Advanced formation detection with player role identification"""
        try:
            # Get player positions
            positions = np.array([list(p['position']) for p in players])
            
            if len(positions) < 7:  # Need enough players for meaningful analysis
                return "Unknown", 0.4, positions.tolist()
            
            # Determine field orientation (which direction team is attacking)
            # For simplicity, we'll assume left-to-right is the attacking direction
            # This would need to be adjusted based on actual game context
            attacking_left_to_right = True
            
            # Normalize positions to 0-1 range for better clustering
            x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
            y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
            
            range_x = x_max - x_min if x_max > x_min else 1
            range_y = y_max - y_min if y_max > y_min else 1
            
            normalized_positions = positions.copy()
            normalized_positions[:, 0] = (positions[:, 0] - x_min) / range_x
            normalized_positions[:, 1] = (positions[:, 1] - y_min) / range_y
            
            # Identify goalkeeper (usually the player furthest back)
            gk_idx = np.argmin(normalized_positions[:, 0]) if attacking_left_to_right else np.argmax(normalized_positions[:, 0])
            
            # Remove goalkeeper from formation analysis
            outfield_positions = np.delete(normalized_positions, gk_idx, axis=0)
            original_positions = np.delete(positions, gk_idx, axis=0)
            
            # Apply DBSCAN clustering to potentially identify lines
            db = DBSCAN(eps=0.15, min_samples=2)
            clusters = db.fit_predict(outfield_positions)
            
            # If DBSCAN fails to find good clusters, fall back to K-means
            if len(np.unique(clusters[clusters >= 0])) < 2:
                # Try K-means with variable number of clusters
                best_inertia = float('inf')
                best_k = 3
                best_labels = None
                
                for k in range(3, 6):  # Try 3, 4, and 5 clusters
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(outfield_positions)
                    
                    if kmeans.inertia_ < best_inertia:
                        best_inertia = kmeans.inertia_
                        best_k = k
                        best_labels = kmeans.labels_
                
                clusters = best_labels
            
            # Count players in each cluster or line
            unique_clusters = np.unique(clusters)
            cluster_counts = []
            
            for c in unique_clusters:
                if c >= 0:  # Ignore noise points labeled as -1
                    count = np.sum(clusters == c)
                    cluster_counts.append((c, count))
            
            # Sort clusters by average x-position (for attacking direction)
            cluster_x_means = []
            for c, _ in cluster_counts:
                mean_x = np.mean(outfield_positions[clusters == c, 0])
                cluster_x_means.append((c, mean_x))
            
            # Sort from defense to attack
            if attacking_left_to_right:
                cluster_x_means.sort(key=lambda x: x[1])
            else:
                cluster_x_means.sort(key=lambda x: -x[1])
            
            # Organize players into lines
            lines = []
            for c, _ in cluster_x_means:
                line_size = np.sum(clusters == c)
                lines.append(int(line_size))
            
            # Append goalkeeper to the formation
            lines = [1] + lines  # Add goalkeeper as "1"
            
            # Limit to common football formations with sanity checks
            if len(lines) < 3:
                # Not enough lines detected, fallback to simpler method
                formation, conf = self.calculate_formation_clustering(players)
                return formation, conf, positions.tolist()
            
            # Convert to standard formation string (ignoring goalkeeper)
            formation_str = "-".join(str(l) for l in lines[1:])
            
            # Calculate confidence based on clustering quality
            if len(np.unique(clusters)) <= 1:
                confidence = 0.5  # Low confidence if clustering failed
            else:
                # Higher confidence with more distinct lines
                confidence = min(0.9, 0.6 + 0.1 * len(np.unique(clusters)))
            
            # Map to common formations if close
            common_formations = {
                "4-4-2": 0,
                "4-3-3": 0,
                "3-5-2": 0,
                "5-3-2": 0,
                "4-2-3-1": 0,
                "3-4-3": 0
            }
            
            for common in common_formations:
                common_parts = common.split("-")
                if len(common_parts) == len(lines) - 1:
                    similarity = sum(abs(int(common_parts[i]) - lines[i+1]) for i in range(len(common_parts)))
                    if similarity <= 2:  # Allow small variations
                        formation_str = common
                        confidence = max(confidence, 0.8)
                        break
            
            return formation_str, confidence, positions.tolist()
            
        except Exception as e:
            print(f"Enhanced formation detection error: {str(e)}")
            traceback.print_exc()
            # Fallback to clustering method
            formation, conf = self.calculate_formation_clustering(players)
            return formation, conf, positions.tolist()
    
    def draw_formation_lines(self, frame, players, color):
        """Draw lines connecting players in the same line (defense, midfield, attack)"""
        try:
            # Sort players by y-coordinate (vertical position)
            sorted_players = sorted(players, key=lambda p: p['position'][1])
            
            # Group players into lines
            y_positions = [p['position'][1] for p in sorted_players]
            y_min, y_max = min(y_positions), max(y_positions)
            range_y = y_max - y_min if y_max > y_min else 1
            
            defenders = [p for p in sorted_players if (p['position'][1] - y_min) / range_y < 0.33]
            midfielders = [p for p in sorted_players if 0.33 <= (p['position'][1] - y_min) / range_y < 0.66]
            attackers = [p for p in sorted_players if (p['position'][1] - y_min) / range_y >= 0.66]
            
            # Sort each line by x-coordinate
            for line in [defenders, midfielders, attackers]:
                if len(line) > 1:
                    line.sort(key=lambda p: p['position'][0])
                    
                    # Draw lines connecting players in the same line
                    for i in range(len(line) - 1):
                        pt1 = (int(line[i]['position'][0]), int(line[i]['position'][1]))
                        pt2 = (int(line[i+1]['position'][0]), int(line[i+1]['position'][1]))
                        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
                    
                    # Draw line name and role
                    if line:
                        line_type = "DEF" if line == defenders else "MID" if line == midfielders else "ATT"
                        avg_x = sum(p['position'][0] for p in line) / len(line)
                        avg_y = sum(p['position'][1] for p in line) / len(line)
                        cv2.putText(frame, f"{line_type} ({len(line)})", 
                                   (int(avg_x), int(avg_y - 15)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(f"Error drawing formation lines: {str(e)}")
    
    def detect_events(self, frame_idx, players, ball_position):
        """Detect key events like passes and shots with enhanced accuracy"""
        try:
            # Need at least a few frames of ball positions
            if len(self.ball_positions) < 5:
                return
            
            # Calculate ball velocity vector
            prev_ball_positions = list(self.ball_positions)[-5:]
            ball_vector = (
                ball_position[0] - prev_ball_positions[0][0],
                ball_position[1] - prev_ball_positions[0][1]
            )
            ball_speed = np.sqrt(ball_vector[0]**2 + ball_vector[1]**2)
            
            # Get nearest players to current and previous ball positions
            if players:
                current_nearest = min(players, key=lambda p: 
                                    distance.euclidean(p['position'], ball_position))
                prev_nearest = min(players, key=lambda p: 
                                  distance.euclidean(p['position'], prev_ball_positions[0]))
                
                # Enhanced pass detection with sensitivity setting
                pass_speed_threshold = 5 * (self.pass_detection_sensitivity / 5)  # Adjust based on sensitivity
                player_distance_threshold = 5 * (10 / self.pass_detection_sensitivity)  # Inverse relationship
                
                # Detect pass - ball moved between players of same team
                if (current_nearest['id'] != prev_nearest['id'] and 
                    current_nearest['team'] == prev_nearest['team'] and 
                    ball_speed > pass_speed_threshold and
                    distance.euclidean(current_nearest['position'], prev_nearest['position']) > player_distance_threshold):
                    
                    # Calculate pass length in meters
                    pixel_to_meter = (FIELD_WIDTH / self.video_info.target_width + FIELD_HEIGHT / self.video_info.target_height) / 2
                    pass_length = distance.euclidean(ball_position, prev_ball_positions[0]) * pixel_to_meter
                    
                    # Determine pass zones
                    zone_from_x = min(int(prev_ball_positions[0][0] / self.video_info.target_width * 9), 8)
                    zone_from_y = min(int(prev_ball_positions[0][1] / self.video_info.target_height * 6), 5)
                    
                    zone_to_x = min(int(ball_position[0] / self.video_info.target_width * 9), 8)
                    zone_to_y = min(int(ball_position[1] / self.video_info.target_height * 6), 5)
                    
                    # Determine pass direction (forward, backward, lateral)
                    dx = ball_position[0] - prev_ball_positions[0][0]
                    dy = ball_position[1] - prev_ball_positions[0][1]
                    
                    # Simplify direction based on dominant axis
                    if abs(dx) > abs(dy):
                        direction = "forward" if dx > 0 else "backward"
                    else:
                        direction = "lateral"
                    
                    # Determine pass type based on trajectory and speed
                    if ball_speed > 20:
                        pass_type = "through" if direction == "forward" else "long"
                    elif abs(ball_vector[1]) > abs(ball_vector[0]) * 2:
                        pass_type = "lofted"
                    else:
                        pass_type = "ground"
                    
                    # Determine if pass is progressive (moves ball significantly forward)
                    progressive = direction == "forward" and dx > self.video_info.target_width / 5
                    
                    # Determine if pass is into danger zone (final third or penalty area)
                    danger_zone = False
                    field_third = self.video_info.target_width / 3
                    
                    # If passing into final third or penalty area
                    if (zone_to_x >= 6 or  # Final third of pitch
                        (zone_to_x >= 7 and zone_to_y >= 2 and zone_to_y <= 3)):  # Penalty area
                        danger_zone = True
                    
                    # Determine if pass breaks defensive lines (simplified)
                    # Find players from opposite team between passer and receiver
                    opp_team = self.team_away if current_nearest['team'] == self.team_home else self.team_home
                    opp_players = [p for p in players if p['team'] == opp_team]
                    
                    # Check if pass goes between/through opposing players
                    breaking_lines = False
                    if opp_players:
                        # Create a simplified line from passer to receiver
                        pass_dir_x = ball_position[0] - prev_ball_positions[0][0]
                        pass_dir_y = ball_position[1] - prev_ball_positions[0][1]
                        pass_length_pixels = np.sqrt(pass_dir_x**2 + pass_dir_y**2)
                        
                        # Check if any opposing players are near the pass line but not too close to passer/receiver
                        for opp in opp_players:
                            # Distance from opponent to pass line
                            d = self.point_to_line_distance(
                                opp['position'], 
                                prev_ball_positions[0], 
                                ball_position
                            )
                            
                            # Check if opponent is near the pass line
                            if d < 30:  # Pixels threshold
                                breaking_lines = True
                                break
                    
                    # Determine if pass is a switch of play (cross-field pass)
                    switch_play = abs(dy) > self.video_info.target_height / 3
                    
                    # Calculate expected assists (xA) - simplified model
                    # Higher xA for passes into dangerous areas or that create shooting opportunities
                    xA = 0.0
                    if danger_zone:
                        if pass_type == "through":
                            xA = 0.15  # Through passes into danger zone have higher xA
                        else:
                            xA = 0.05  # Other passes into danger zone
                    elif progressive:
                        xA = 0.02  # Progressive passes have some xA value
                    
                    # Track pass length distribution for analysis
                    # Categorize passes as short, medium, or long
                    if pass_length < 10:
                        length_category = "short"
                    elif pass_length < 25:
                        length_category = "medium"
                    else:
                        length_category = "long"
                    
                    self.pass_length_distribution[current_nearest['team']][length_category] += 1
                    
                    # Create pass event
                    pass_event = PassEvent(
                        time=frame_idx / self.video_info.fps,
                        from_player=prev_nearest['id'],
                        to_player=current_nearest['id'],
                        team=current_nearest['team'],
                        from_position=prev_ball_positions[0],
                        to_position=ball_position,
                        completed=True,
                        length=pass_length,
                        zone_from=(zone_from_x, zone_from_y),
                        zone_to=(zone_to_x, zone_to_y),
                        direction=direction,
                        pass_type=pass_type,
                        progressive=progressive,
                        danger_zone=danger_zone,
                        breaking_lines=breaking_lines,
                        switch_play=switch_play,
                        xA=xA
                    )
                    
                    # Add to pass data
                    self.pass_data.append(asdict(pass_event))
                    
                    # Update pass statistics
                    self.pass_types[current_nearest['team']][pass_type] += 1
                    self.pass_directions[current_nearest['team']][direction] += 1
                    
                    # Track progressive passes and danger zone passes
                    if progressive:
                        self.progressive_passes[current_nearest['team']] += 1
                    if danger_zone:
                        self.danger_zone_passes[current_nearest['team']] += 1
                    if breaking_lines:
                        self.breaking_lines_passes[current_nearest['team']] += 1
                    if switch_play:
                        self.switch_play_passes[current_nearest['team']] += 1
                    
                    # Add xA to total
                    self.total_xA[current_nearest['team']] += xA
                    
                    # Update pass network
                    self.pass_networks[current_nearest['team']][prev_nearest['id']][current_nearest['id']] += 1
                    
                    # Update zone statistics
                    team_idx = 0 if current_nearest['team'] == self.team_home else 1
                    self.zone_passes[zone_from_y, zone_from_x, team_idx] += 1
                    
                    # Add to general events list
                    self.events.append({
                        'time': frame_idx / self.video_info.fps,
                        'type': 'pass',
                        'from_player': prev_nearest['id'],
                        'to_player': current_nearest['id'],
                        'team': current_nearest['team'],
                        'pass_type': pass_type,
                        'direction': direction,
                        'progressive': progressive,
                        'danger_zone': danger_zone,
                        'breaking_lines': breaking_lines,
                        'switch_play': switch_play,
                        'length': pass_length,
                        'xA': xA
                    })
                
                # Enhanced shot detection with improved sensitivity
                goal_line_left = 0
                goal_line_right = self.video_info.target_width
                goal_center_left = (goal_line_left, self.video_info.target_height / 2)
                goal_center_right = (goal_line_right, self.video_info.target_height / 2)
                
                # Calculate shot threshold based on sensitivity
                shot_speed_threshold = 15 * (self.shot_detection_sensitivity / 5)
                shot_angle_threshold = 30 * (10 / self.shot_detection_sensitivity)
                
                # Check if ball is moving toward either goal with high speed
                angle_to_left = self.angle_between_vectors(
                    ball_vector, 
                    (goal_center_left[0] - ball_position[0], goal_center_left[1] - ball_position[1])
                )
                
                angle_to_right = self.angle_between_vectors(
                    ball_vector, 
                    (goal_center_right[0] - ball_position[0], goal_center_right[1] - ball_position[1])
                )
                
                is_shot_left = ball_speed > shot_speed_threshold and angle_to_left < shot_angle_threshold
                is_shot_right = ball_speed > shot_speed_threshold and angle_to_right < shot_angle_threshold
                
                if is_shot_left or is_shot_right:
                    target_goal = self.team_away if is_shot_left else self.team_home
                    
                    # Calculate shot distance from goal
                    goal_position = goal_center_left if is_shot_left else goal_center_right
                    pixel_to_meter = (FIELD_WIDTH / self.video_info.target_width + FIELD_HEIGHT / self.video_info.target_height) / 2
                    shot_distance = distance.euclidean(prev_ball_positions[0], goal_position) * pixel_to_meter
                    
                    # Calculate shot angle from goal center
                    shot_angle = self.angle_between_vectors(
                        (goal_position[0] - prev_ball_positions[0][0], goal_position[1] - prev_ball_positions[0][1]),
                        (1, 0) if is_shot_left else (-1, 0)  # Perpendicular to goal line
                    )
                    
                    # Determine if shot is on target (simplified)
                    on_target = ball_speed > 25 and shot_angle < 15
                    
                    # Calculate expected goal (xG) value based on distance and angle
                    # Enhanced xG model with distance and angle factors
                    distance_factor = 1 / (1 + (shot_distance/10)**2)  # Decay with square of distance
                    angle_factor = 1 - (min(shot_angle, 90) / 90)**2  # Penalty for wide angles
                    
                    # Baseline xG
                    xg_baseline = 0.3 * distance_factor * angle_factor
                    
                    # Adjust for location
                    penalty_box = shot_distance < 18 and shot_angle < 45
                    six_yard_box = shot_distance < 6 and shot_angle < 30
                    
                    if six_yard_box:
                        xg_location_factor = 2.5  # Much higher xG in six-yard box
                    elif penalty_box:
                        xg_location_factor = 1.5  # Higher xG in penalty box
                    else:
                        xg_location_factor = 1.0
                    
                    # Final xG calculation with bounds
                    xg = min(0.9, max(0.01, xg_baseline * xg_location_factor))
                    
                    # Determine shot type based on ball position and speed
                    shot_type = "normal"
                    if ball_position[1] < (prev_ball_positions[0][1] - 20):  # Ball moved upward significantly
                        shot_type = "volley"
                    
                    # Get other nearby players to determine pressure
                    other_players = [p for p in players if p['id'] != prev_nearest['id']]
                    if other_players:
                        min_distance = min(distance.euclidean(p['position'], prev_ball_positions[0]) for p in other_players)
                        # Normalize pressure from 0-1 based on distance
                        pressure = max(0, min(1.0, 30.0 / max(min_distance, 1.0)))
                    else:
                        pressure = 0
                    
                    # Determine shot scenario
                    # For simplicity, we'll use simple heuristics
                    scenario = "open_play"
                    
                    # Determine shot zone
                    shot_position = prev_ball_positions[0]
                    center_y = self.video_info.target_height / 2
                    if abs(shot_position[1] - center_y) < self.video_info.target_height / 6:
                        zone = "central"
                    elif shot_position[1] < center_y:
                        zone = "left_side"
                    else:
                        zone = "right_side"
                    
                    # Further refine with distance
                    if shot_distance < 18:  # Penalty box distance in meters
                        zone = f"{zone}_box"
                    else:
                        zone = f"{zone}_outside_box"
                    
                    # Create shot event
                    shot_event = ShotEvent(
                        time=frame_idx / self.video_info.fps,
                        player=prev_nearest['id'],
                        team=prev_nearest['team'],
                        position=prev_ball_positions[0],target_goal=target_goal,
                        on_target=on_target,
                        goal=False,  # Would need additional detection for goals
                        expected_goal=xg,
                        distance=shot_distance,
                        angle=shot_angle,
                        shot_type=shot_type,
                        scenario=scenario,
                        pressure=pressure,
                        zone=zone
                    )
                    
                    # Add to shot data
                    self.shot_data.append(asdict(shot_event))
                    
                    # Update shot statistics
                    team = prev_nearest['team']
                    self.shot_types[team][shot_type] += 1
                    self.shot_scenarios[team][scenario] += 1
                    self.shot_zones[team][zone] += 1
                    self.total_xG[team] += xg
                    
                    if pressure > 0.5:  # Threshold for "under pressure"
                        self.shots_under_pressure[team] += 1
                    
                    # Update zone statistics
                    zone_x = min(int(prev_ball_positions[0][0] / self.video_info.target_width * 9), 8)
                    zone_y = min(int(prev_ball_positions[0][1] / self.video_info.target_height * 6), 5)
                    team_idx = 0 if team == self.team_home else 1
                    self.zone_shots[zone_y, zone_x, team_idx] += 1
                    
                    # Add to general events list
                    self.events.append({
                        'time': frame_idx / self.video_info.fps,
                        'type': 'shot',
                        'player': prev_nearest['id'],
                        'team': prev_nearest['team'],
                        'target_goal': target_goal,
                        'on_target': on_target,
                        'xG': xg,
                        'shot_type': shot_type,
                        'scenario': scenario,
                        'pressure': pressure,
                        'zone': zone,
                        'distance': shot_distance,
                        'angle': shot_angle
                    })
        except Exception as e:
            print(f"Error in event detection: {str(e)}")
            traceback.print_exc()

    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate the shortest distance from a point to a line segment"""
        x1, y1 = line_start
        x2, y2 = line_end
        x0, y0 = point
        
        # Length of line segment squared
        l2 = (x2 - x1)**2 + (y2 - y1)**2
        if l2 == 0:  # Line segment is a point
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Consider the line extending the segment, parameterized as start + t (end - start)
        # Find projection of point onto the line
        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / l2))
        
        # Calculate the projection point
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        # Distance from point to projection
        return np.sqrt((x0 - proj_x)**2 + (y0 - proj_y)**2)
    
    def angle_between_vectors(self, v1, v2):
        """Calculate angle between two vectors in degrees"""
        try:
            v1_norm = np.sqrt(v1[0]**2 + v1[1]**2)
            v2_norm = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if v1_norm == 0 or v2_norm == 0:
                return 0
                
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            cos_angle = dot_product / (v1_norm * v2_norm)
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle_rad)
        except Exception as e:
            print(f"Error calculating angle: {str(e)}")
            return 0
    
    def process_video(self):
        """Process the video with player detection and tracking"""
        try:
            # Initialize model if not already done
            if not hasattr(self, 'model'):
                self.initialize_model()
                
            cap, temp_video_path = self.preprocess_video(self.video_path)
            if cap is None:
                st.error("Error loading video file.")
                return None
            
            # Create output video writer
            output_video_path = "enhanced_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, self.video_info.fps, 
                                 (self.video_info.target_width, self.video_info.target_height))
            
            # Set up progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process frames
            frames_to_process = min(self.video_info.total_frames, self.max_frames_to_process)
            frame_count = 0
            processed = 0
            
            while cap.isOpened() and frame_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames for faster processing
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Process this frame
                processed_frame = self.analyze_frame(frame, frame_count)
                
                # Write processed frame to output video
                out.write(processed_frame)
                
                # Update progress
                processed += 1
                progress = min(processed / (frames_to_process / self.frame_skip), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{frames_to_process} ({progress*100:.1f}%)")
                
                frame_count += 1
                self.video_info.processed_frames += 1
                
            # Release resources
            cap.release()
            out.release()
            
            # Clean up temp file
            try:
                os.unlink(temp_video_path)
            except:
                pass
            
            # Save video path to session state
            st.session_state.processed_video_path = output_video_path
            
            # Update video info in session state
            st.session_state[KEY_VIDEO_INFO] = self.video_info
            
            # Prepare analysis results
            self.prepare_analysis_results()
            
            # Generate AI insights if enabled
            if self.enable_gemini and st.session_state.gemini_api_key:
                self.generate_gemini_insights()
            
            # Set analysis complete flag in session state
            st.session_state[KEY_ANALYSIS_COMPLETE] = True
            
            # Return path to processed video
            return output_video_path
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            traceback.print_exc()
            return None
            
    def prepare_analysis_results(self):
        """Prepare analysis results for display"""
        try:
            # Calculate player statistics
            player_stats = {}
            for player_id in self.player_positions.keys():
                if player_id not in self.player_team:
                    continue
                    
                team = self.player_team[player_id]
                
                # Calculate average speed
                avg_speed = np.mean(self.speed_data[player_id]) if self.speed_data[player_id] else 0
                max_speed = np.max(self.speed_data[player_id]) if self.speed_data[player_id] else 0
                
                # Calculate possession percentage
                total_frames = max(sum(self.team_possession_frames.values()), 1)  # Avoid division by zero
                possession_percentage = (self.ball_possession[player_id] / total_frames * 100) if total_frames > 0 else 0
                
                # Enhanced pass statistics
                passes_completed = sum(1 for p in self.pass_data if p['from_player'] == player_id)
                passes_received = sum(1 for p in self.pass_data if p['to_player'] == player_id)
                
                # Pass completion rate
                pass_completion_rate = 100  # Default to 100% if no data
                
                # Calculate player influence (based on pass network centrality - simplified)
                player_influence = passes_completed + passes_received
                
                # Calculate progressive passes
                progressive_passes = sum(1 for p in self.pass_data if p['from_player'] == player_id and p.get('progressive', False))
                
                # Calculate breaking lines passes
                breaking_lines_passes = sum(1 for p in self.pass_data if p['from_player'] == player_id and p.get('breaking_lines', False))
                
                # Calculate expected assists (xA)
                player_xA = sum(p.get('xA', 0) for p in self.pass_data if p['from_player'] == player_id)
                
                # Enhanced shot statistics
                shots = sum(1 for s in self.shot_data if s['player'] == player_id)
                shots_on_target = sum(1 for s in self.shot_data if s['player'] == player_id and s.get('on_target', False))
                
                # Expected goals (xG)
                player_xG = sum(s.get('expected_goal', 0) for s in self.shot_data if s['player'] == player_id)
                
                # Store player stats
                player_stats[player_id] = {
                    'Player ID': player_id,
                    'Team': team,
                    'Distance (m)': round(self.distance_data[player_id], 2),
                    'Avg Speed (m/s)': round(avg_speed, 2),
                    'Max Speed (m/s)': round(max_speed, 2),
                    'Possession (%)': round(possession_percentage, 2),
                    'Passes': passes_completed,
                    'Passes Received': passes_received,
                    'Pass Completion (%)': round(pass_completion_rate, 2),
                    'Progressive Passes': progressive_passes,
                    'Breaking Lines Passes': breaking_lines_passes,
                    'Expected Assists (xA)': round(player_xA, 3),
                    'Shots': shots,
                    'Shots on Target': shots_on_target,
                    'Expected Goals (xG)': round(player_xG, 3),
                    'Influence': player_influence
                }
            
            # Store in session state
            st.session_state[KEY_PLAYER_STATS] = player_stats
            
            # Create DataFrame for visualization
            self.player_stats_df = pd.DataFrame.from_dict(player_stats, orient='index') if player_stats else pd.DataFrame()
            
            # Calculate team statistics
            total_frames = max(sum(self.team_possession_frames.values()), 1)  # Avoid division by zero
            
            # Calculate pass success rate
            for team in [self.team_home, self.team_away]:
                total_passes = sum(1 for p in self.pass_data if p['team'] == team)
                completed_passes = sum(1 for p in self.pass_data if p['team'] == team and p.get('completed', True))
                self.pass_success_rate[team] = (completed_passes / total_passes * 100) if total_passes > 0 else 0
            
            # Calculate shot success rate
            for team in [self.team_home, self.team_away]:
                total_shots = sum(1 for s in self.shot_data if s['team'] == team)
                shots_on_target = sum(1 for s in self.shot_data if s['team'] == team and s.get('on_target', True))
                self.shot_success_rate[team] = (shots_on_target / total_shots * 100) if total_shots > 0 else 0
            
            # Enhanced team stats
            self.team_stats = {
                self.team_home: {
                    'Possession (%)': round(self.team_possession_frames[self.team_home] / total_frames * 100, 2) if total_frames > 0 else 0,
                    'Distance (m)': round(sum(self.distance_data[p] for p, team in self.player_team.items() if team == self.team_home), 2),
                    'Passes': sum(1 for p in self.pass_data if p['team'] == self.team_home),
                    'Pass Completion (%)': round(self.pass_success_rate[self.team_home], 2),
                    'Forward Passes (%)': round(sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('direction') == 'forward') / 
                                 max(sum(1 for p in self.pass_data if p['team'] == self.team_home), 1) * 100, 2),
                    'Progressive Passes': self.progressive_passes[self.team_home],
                    'Breaking Lines Passes': self.breaking_lines_passes[self.team_home],
                    'Danger Zone Passes': self.danger_zone_passes[self.team_home],
                    'Expected Assists (xA)': round(self.total_xA[self.team_home], 3),
                    'Shots': sum(1 for s in self.shot_data if s['team'] == self.team_home),
                    'Shots on Target': sum(1 for s in self.shot_data if s['team'] == self.team_home and s.get('on_target', False)),
                    'Shot Accuracy (%)': round(self.shot_success_rate[self.team_home], 2),
                    'Shots Under Pressure': self.shots_under_pressure[self.team_home],
                    'Expected Goals (xG)': round(self.total_xG[self.team_home], 3),
                    'Most Used Formation': max(self.team_formations[self.team_home].items(), key=lambda x: x[1])[0] if self.team_formations[self.team_home] else "N/A"
                },
                self.team_away: {
                    'Possession (%)': round(self.team_possession_frames[self.team_away] / total_frames * 100, 2) if total_frames > 0 else 0,
                    'Distance (m)': round(sum(self.distance_data[p] for p, team in self.player_team.items() if team == self.team_away), 2),
                    'Passes': sum(1 for p in self.pass_data if p['team'] == self.team_away),
                    'Pass Completion (%)': round(self.pass_success_rate[self.team_away], 2),
                    'Forward Passes (%)': round(sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('direction') == 'forward') / 
                                 max(sum(1 for p in self.pass_data if p['team'] == self.team_away), 1) * 100, 2),
                    'Progressive Passes': self.progressive_passes[self.team_away],
                    'Breaking Lines Passes': self.breaking_lines_passes[self.team_away],
                    'Danger Zone Passes': self.danger_zone_passes[self.team_away],
                    'Expected Assists (xA)': round(self.total_xA[self.team_away], 3),
                    'Shots': sum(1 for s in self.shot_data if s['team'] == self.team_away),
                    'Shots on Target': sum(1 for s in self.shot_data if s['team'] == self.team_away and s.get('on_target', False)),
                    'Shot Accuracy (%)': round(self.shot_success_rate[self.team_away], 2),
                    'Shots Under Pressure': self.shots_under_pressure[self.team_away],
                    'Expected Goals (xG)': round(self.total_xG[self.team_away], 3),
                    'Most Used Formation': max(self.team_formations[self.team_away].items(), key=lambda x: x[1])[0] if self.team_formations[self.team_away] else "N/A"
                }
            }
            
            # Store team stats in session state
            st.session_state[KEY_TEAM_STATS] = self.team_stats
            
            # Create zones data for heatmap
            total_zone_time = np.sum(self.zone_possession)
            if total_zone_time > 0:
                self.zone_percentage = self.zone_possession / total_zone_time * 100
            else:
                self.zone_percentage = np.zeros_like(self.zone_possession)
                
            # Store events in session state
            st.session_state[KEY_EVENTS] = self.events
        except Exception as e:
            st.error(f"Error preparing analysis results: {str(e)}")
            traceback.print_exc()
    
    def display_video_analysis(self, output_video_path):
        """Display processed video and basic analysis"""
        try:
            with self.tab1:
                st.subheader("📹 Processed Video with Player Tracking")
                if os.path.exists(output_video_path):
                    st.video(output_video_path)
                else:
                    st.error("Video file not found. The analysis may have failed.")
                    return
                
                # Display basic match info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"⚔️ {self.team_home} vs {self.team_away}")
                    st.write(f"Total Frames Analyzed: {self.video_info.processed_frames}")
                    st.write(f"Video Length: {self.video_info.duration:.2f} seconds")
                    st.write(f"Analyzed Players: {len(self.player_positions)}")
                
                with col2:
                    # Display possession pie chart
                    if sum(self.team_possession_frames.values()) > 0:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        labels = [self.team_home, self.team_away]
                        sizes = [self.team_stats[self.team_home]['Possession (%)'], 
                                self.team_stats[self.team_away]['Possession (%)']]
                        colors = [self.home_color, self.away_color]
                        
                        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                        ax.axis('equal')
                        plt.title('Ball Possession')
                        st.pyplot(fig)
                
                # Display events timeline
                if self.events:
                    st.subheader("⏱️ Key Events Timeline")
                    
                    # Group events into expandable sections by 15-minute intervals
                    events_by_interval = {}
                    interval_size = 15 * 60  # 15 minutes in seconds
                    
                    for event in sorted(self.events, key=lambda e: e['time']):
                        interval = int(event['time'] // interval_size)
                        interval_str = f"{interval * 15}-{(interval + 1) * 15} min"
                        
                        if interval_str not in events_by_interval:
                            events_by_interval[interval_str] = []
                        
                        events_by_interval[interval_str].append(event)
                    
                    # Display events by interval in expandable sections
                    for interval_str, interval_events in events_by_interval.items():
                        with st.expander(f"Events ({interval_str})"):
                            for event in interval_events:
                                time_str = f"{int(event['time'] // 60)}:{int(event['time'] % 60):02d}"
                                team = event['team']
                                team_color = self.home_color if team == self.team_home else self.away_color
                                
                                if event['type'] == 'pass':
                                    # Enhanced pass description
                                    direction = event.get('direction', 'unknown')
                                    pass_type = event.get('pass_type', 'normal')
                                    progressive = "progressive " if event.get('progressive', False) else ""
                                    danger_zone = "into danger zone " if event.get('danger_zone', False) else ""
                                    breaking_lines = "breaking lines " if event.get('breaking_lines', False) else ""
                                    switch_play = "switch play " if event.get('switch_play', False) else ""
                                    
                                    pass_quality = ""
                                    if event.get('xA', 0) > 0.1:
                                        pass_quality = "high-quality "
                                    
                                    st.markdown(f"<span style='color:{team_color}'>⏱️ {time_str} - **{pass_quality}{progressive}{breaking_lines}{danger_zone}{switch_play}{pass_type} {direction} pass** from Player {event['from_player']} to Player {event['to_player']} ({team})</span>", unsafe_allow_html=True)
                                
                                elif event['type'] == 'shot':
                                    # Enhanced shot description
                                    on_target = "on target" if event.get('on_target', False) else "off target"
                                    xg = f" (xG: {event.get('xG', 0):.2f})" if 'xG' in event else ""
                                    shot_type = event.get('shot_type', 'normal')
                                    zone = event.get('zone', '')
                                    pressure = "under pressure " if event.get('pressure', 0) > 0.5 else ""
                                    distance = f"from {event.get('distance', 0):.1f}m " if 'distance' in event else ""
                                    
                                    st.markdown(f"<span style='color:{team_color}'>⏱️ {time_str} - **{shot_type.capitalize()} shot {on_target}** {pressure}{distance}by Player {event['player']} ({team}) at {event['target_goal']} goal from {zone}{xg}</span>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying video analysis: {str(e)}")
            traceback.print_exc()
    
    def display_player_stats(self):
        """Display detailed player statistics"""
        try:
            with self.tab2:
                st.subheader("🔍 Player Statistics")
                
                # Check if player stats are available
                if self.player_stats_df.empty:
                    st.info("No player statistics available. Run the analysis first.")
                    return
                
                # Filter options
                team_filter = st.radio("Filter by Team", ["All", self.team_home, self.team_away], horizontal=True)
                
                # Apply filters
                if team_filter != "All":
                    filtered_df = self.player_stats_df[self.player_stats_df['Team'] == team_filter]
                else:
                    filtered_df = self.player_stats_df
                
                # Sort options
                sort_options = ["Distance (m)", "Avg Speed (m/s)", "Max Speed (m/s)", 
                                "Possession (%)", "Passes", "Passes Received", 
                                "Pass Completion (%)", "Progressive Passes", "Breaking Lines Passes",
                                "Expected Assists (xA)", "Shots", "Shots on Target", 
                                "Expected Goals (xG)", "Influence"]
                
                # Make sure sort_by is a column in the DataFrame
                available_columns = filtered_df.columns.tolist()
                sort_options = [col for col in sort_options if col in available_columns]
                
                sort_by = st.selectbox("Sort by", sort_options) if sort_options else "Distance (m)"
                
                # Display sorted table
                if sort_by in filtered_df.columns:
                    st.dataframe(filtered_df.sort_values(by=sort_by, ascending=False))
                else:
                    st.dataframe(filtered_df)
                
                # Display player distance comparison
                st.subheader("🏃 Player Distance Comparison")
                
                if 'Distance (m)' in filtered_df.columns and not filtered_df.empty:
                    fig = px.bar(filtered_df.sort_values(by="Distance (m)", ascending=False).head(10),
                                x="Player ID", y="Distance (m)", color="Team",
                                color_discrete_map={self.team_home: self.home_color, self.team_away: self.away_color})
                    
                    fig.update_layout(title="Top 10 Players by Distance Covered")
                    st.plotly_chart(fig)
                else:
                    st.info("Distance data not available.")
                
                # Display attacking contributions visualization
                st.subheader("🎯 Player Attacking Contributions")
                
                # Create contributing metrics visualization if data available
                if not filtered_df.empty and 'Expected Goals (xG)' in filtered_df.columns and 'Expected Assists (xA)' in filtered_df.columns:
                    # Filter for players with significant contributions
                    contrib_df = filtered_df[
                        (filtered_df['Expected Goals (xG)'] > 0.05) | 
                        (filtered_df['Expected Assists (xA)'] > 0.05)
                    ].copy()
                    
                    if not contrib_df.empty:
                        # Add total goal contribution
                        contrib_df['Goal Contribution'] = contrib_df['Expected Goals (xG)'] + contrib_df['Expected Assists (xA)']
                        
                        # Create scatter plot
                        fig = px.scatter(
                            contrib_df, 
                            x="Expected Goals (xG)", 
                            y="Expected Assists (xA)", 
                            size="Goal Contribution",
                            color="Team",
                            color_discrete_map={self.team_home: self.home_color, self.team_away: self.away_color},
                            hover_name="Player ID",
                            text="Player ID",
                            size_max=30,
                            title="Expected Goals vs Expected Assists"
                        )
                        
                        # Update layout
                        fig.update_layout(
                            xaxis_title="Expected Goals (xG)",
                            yaxis_title="Expected Assists (xA)"
                        )
                        
                        st.plotly_chart(fig)
                    else:
                        st.info("Not enough attacking contribution data available.")
                else:
                    st.info("Expected goals and assists data not available.")
                
                # Display player speed profiles
                st.subheader("⚡ Player Speed Profiles")
                
                # Allow selecting players to display
                if not filtered_df.empty:
                    selected_players = st.multiselect("Select Players for Speed Analysis", 
                                                    options=filtered_df.index.tolist(),
                                                    default=filtered_df.sort_values(by="Max Speed (m/s)", ascending=False).head(3).index.tolist() if "Max Speed (m/s)" in filtered_df.columns else [])
                    
                    if selected_players:
                        speed_data = {}
                        for player_id in selected_players:
                            if player_id in self.speed_data and self.speed_data[player_id]:
                                speed_data[f"Player {player_id} ({self.player_team[player_id]})"] = self.speed_data[player_id]
                        
                        if speed_data:
                            fig = go.Figure()
                            
                            for player, speeds in speed_data.items():
                                team = player.split('(')[1].split(')')[0]
                                color = self.home_color if team == self.team_home else self.away_color
                                
                                # Smooth data with moving average
                                smoothed_speeds = pd.Series(speeds).rolling(window=5, min_periods=1).mean().values
                                
                                fig.add_trace(go.Scatter(
                                    y=smoothed_speeds,
                                    mode='lines',
                                    name=player,
                                    line=dict(color=color)
                                ))
                            
                            fig.update_layout(title="Player Speed Over Time",
                                            xaxis_title="Frame Number",
                                            yaxis_title="Speed (m/s)")
                            
                            st.plotly_chart(fig)
                        else:
                            st.info("No speed data available for selected players.")
                    else:
                        st.info("Select players to display speed profiles.")
                else:
                    st.info("No player data available.")
        except Exception as e:
            st.error(f"Error displaying player stats: {str(e)}")
            traceback.print_exc()
    
    def display_spatial_analysis(self):
        """Display spatial analysis like heatmaps and player movement"""
        try:
            with self.tab3:
                st.subheader("🌍 Spatial Analysis")
                
                # Player movement heatmap
                st.subheader("🔥 Player Movement Heatmaps")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### {self.team_home} Team Heatmap")
                    self.generate_team_heatmap(self.team_home)
                
                with col2:
                    st.markdown(f"#### {self.team_away} Team Heatmap")
                    self.generate_team_heatmap(self.team_away)
                
                # Zone control analysis
                st.subheader("🎮 Pitch Zone Control")
                
                if hasattr(self, 'zone_percentage') and np.sum(self.zone_possession) > 0:
                    # Create heatmap for zone control
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Calculate dominance ratio (home vs away)
                    zone_diff = np.zeros((6, 9))
                    for i in range(6):
                        for j in range(9):
                            home = self.zone_possession[i, j, 0]
                            away = self.zone_possession[i, j, 1]
                            total = home + away
                            
                            if total > 0:
                                zone_diff[i, j] = (home - away) / total  # Range: [-1, 1]
                    
                    # Plot heatmap
                    sns.heatmap(zone_diff, cmap="RdBu_r", vmin=-1, vmax=1, 
                               cbar_kws={'label': f'{self.team_away} <-- Zone Control --> {self.team_home}'},
                               annot=False, ax=ax)
                    
                    ax.set_title("Zone Control Analysis")
                    ax.set_xlabel("Width")
                    ax.set_ylabel("Height")
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    
                    st.pyplot(fig)
                else:
                    st.info("Zone control data not available. Run the analysis first.")
                
                # Pass network analysis
                if self.pass_data:
                    st.subheader("🔄 Pass Network Analysis")
                    
                    team_filter = st.radio("Team Pass Network", [self.team_home, self.team_away], horizontal=True, key="spatial_team_filter")
                    
                    # Filtered pass data
                    team_passes = [p for p in self.pass_data if p['team'] == team_filter]
                    
                    if team_passes:
                        # Create pass network graph
                        G = defaultdict(lambda: defaultdict(int))
                        
                        # Count passes between players
                        for p in team_passes:
                            G[p['from_player']][p['to_player']] += 1
                        
                        # Get average positions
                        avg_positions = {}
                        for player_id in self.player_positions:
                            if player_id in self.player_team and self.player_team[player_id] == team_filter:
                                positions = np.array(self.player_positions[player_id])
                                if len(positions) > 0:
                                    avg_positions[player_id] = np.mean(positions, axis=0)
                        
                        # Create the network visualization
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Draw field
                        rect = plt.Rectangle((0, 0), self.video_info.target_width, self.video_info.target_height, 
                                            facecolor='#238823', alpha=0.3, edgecolor='white')
                        ax.add_patch(rect)
                        
                        # Draw center line and circle
                        ax.plot([self.video_info.target_width/2, self.video_info.target_width/2], [0, self.video_info.target_height], 'white')
                        center_circle = plt.Circle((self.video_info.target_width/2, self.video_info.target_height/2), 
                                                 self.video_info.target_height/10, fill=False, color='white')
                        ax.add_patch(center_circle)
                        
                        # Draw nodes (players)
                        for player_id, pos in avg_positions.items():
                            ax.scatter(pos[0], pos[1], s=150, color=team_filter, edgecolor='black', zorder=2)
                            ax.text(pos[0], pos[1], str(player_id), fontsize=10, 
                                   ha='center', va='center', color='white', fontweight='bold')
                        
                        # Draw edges (passes)
                        for from_p, to_dict in G.items():
                            if from_p in avg_positions:
                                for to_p, weight in to_dict.items():
                                    if to_p in avg_positions:
                                        # Scale width based on number of passes
                                        width = np.log1p(weight) * 1.5
                                        
                                        ax.plot([avg_positions[from_p][0], avg_positions[to_p][0]],
                                               [avg_positions[from_p][1], avg_positions[to_p][1]],
                                               'white', alpha=0.7, linewidth=width, zorder=1)
                        
                        ax.set_xlim(0, self.video_info.target_width)
                        ax.set_ylim(0, self.video_info.target_height)
                        ax.set_title(f"{team_filter} Pass Network")
                        ax.axis('off')
                        
                        st.pyplot(fig)
                        
                        # Enhanced pass analysis
                        st.subheader("📊 Pass Analysis Dashboard")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pass directions chart
                            directions = self.pass_directions[team_filter]
                            
                            if sum(directions.values()) > 0:
                                fig, ax = plt.subplots(figsize=(6, 6))
                                
                                direction_labels = []
                                direction_values = []
                                
                                for d, count in directions.items():
                                    direction_labels.append(d.capitalize())
                                    direction_values.append(count)
                                
                                ax.pie(direction_values, labels=direction_labels, autopct='%1.1f%%', startangle=90)
                                ax.axis('equal')
                                plt.title(f"Pass Directions - {team_filter}")
                                
                                st.pyplot(fig)
                            else:
                                st.info("Not enough pass direction data.")
                        
                        with col2:
                            # Pass types analysis
                            pass_types = self.pass_types[team_filter]
                            
                            if sum(pass_types.values()) > 0:
                                fig = px.bar(
                                    x=list(pass_types.keys()),
                                    y=list(pass_types.values()),
                                    labels={'x': 'Pass Type', 'y': 'Count'},
                                    title=f"Pass Types - {team_filter}",
                                    color_discrete_sequence=[self.home_color if team_filter == self.team_home else self.away_color]
                                )
                                
                                st.plotly_chart(fig)
                            else:
                                st.info("Not enough pass type data.")
                        
                        # Enhanced pass metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            prog_passes = self.progressive_passes[team_filter]
                            st.metric("Progressive Passes", prog_passes)
                        
                        with col2:
                            bl_passes = self.breaking_lines_passes[team_filter]
                            st.metric("Breaking Lines Passes", bl_passes)
                        
                        with col3:
                            dz_passes = self.danger_zone_passes[team_filter]
                            st.metric("Danger Zone Passes", dz_passes)
                        
                        # Pass length distribution
                        st.subheader("📏 Pass Length Distribution")
                        
                        pass_lengths = self.pass_length_distribution[team_filter]
                        if sum(pass_lengths.values()) > 0:
                            # Create horizontal bar chart for pass lengths
                            fig = go.Figure()
                            
                            categories = list(pass_lengths.keys())
                            values = list(pass_lengths.values())
                            
                            fig.add_trace(go.Bar(
                                y=categories,
                                x=values,
                                orientation='h',
                                marker_color=self.home_color if team_filter == self.team_home else self.away_color
                            ))
                            
                            fig.update_layout(
                                title=f"Pass Length Distribution - {team_filter}",
                                xaxis_title="Count",
                                yaxis_title="Length Category"
                            )
                            
                            st.plotly_chart(fig)
                        else:
                            st.info("Not enough pass length data available.")
                        
                    else:
                        st.info(f"No pass data available for {team_filter}.")
                else:
                    st.info("No pass data available. Run the analysis first.")
                
                # Shot analysis
                if self.shot_data:
                    st.subheader("🥅 Shot Analysis")
                    
                    shot_team_filter = st.radio("Team Shot Analysis", [self.team_home, self.team_away], horizontal=True, key="shot_team_filter")
                    
                    # Filtered shot data
                    team_shots = [s for s in self.shot_data if s['team'] == shot_team_filter]
                    
                    if team_shots:
                        # Shot location visualization
                        st.subheader("📍 Shot Locations")
                        
                        fig, ax = plt.subplots(figsize=(10, 7))
                        
                        # Draw field
                        rect = plt.Rectangle((0, 0), self.video_info.target_width, self.video_info.target_height, 
                                            facecolor='#238823', alpha=0.3, edgecolor='white')
                        ax.add_patch(rect)
                        
                        # Draw center line and penalty areas
                        ax.plot([self.video_info.target_width/2, self.video_info.target_width/2], [0, self.video_info.target_height], 'white')
                        
                        # Define penalty area dimensions
                        penalty_width = self.video_info.target_width // 6
                        penalty_height = self.video_info.target_height // 3
                        center_y = self.video_info.target_height // 2
                        
                        # Draw penalty areas
                        ax.add_patch(plt.Rectangle((0, center_y - penalty_height/2), penalty_width, penalty_height, 
                                                 fill=False, edgecolor='white'))
                        ax.add_patch(plt.Rectangle((self.video_info.target_width - penalty_width, center_y - penalty_height/2), 
                                                 penalty_width, penalty_height, fill=False, edgecolor='white'))
                        
                        # Plot shots
                        team_color = self.home_color if shot_team_filter == self.team_home else self.away_color
                        
                        for shot in team_shots:
                            x, y = shot['position']
                            on_target = shot.get('on_target', False)
                            xg = shot.get('expected_goal', 0.05)
                            
                            # Size based on xG, color based on on-target
                            marker_color = 'white' if on_target else 'gray'
                            marker_size = max(50, 200 * xg)  # Scale marker size based on xG
                            
                            ax.scatter(x, y, s=marker_size, color=marker_color, alpha=0.7, edgecolor=team_color, linewidth=2)
                            
                            # Add xG text for significant chances
                            if xg > 0.15:
                                ax.text(x, y, f"{xg:.2f}", fontsize=8, ha='center', va='center', color='black', fontweight='bold')
                        
                        # Set axis limits and hide ticks
                        ax.set_xlim(0, self.video_info.target_width)
                        ax.set_ylim(0, self.video_info.target_height)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # Add legend
                        ax.scatter([], [], s=100, color='white', edgecolor=team_color, label='On Target')
                        ax.scatter([], [], s=100, color='gray', edgecolor=team_color, label='Off Target')
                        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
                        
                        # Add title
                        ax.set_title(f"{shot_team_filter} Shot Map")
                        
                        st.pyplot(fig)
                        
                        # Shot metrics dashboard
                        st.subheader("📊 Shot Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            total_shots = len(team_shots)
                            on_target = sum(1 for s in team_shots if s.get('on_target', False))
                            accuracy = (on_target / total_shots * 100) if total_shots > 0 else 0
                            
                            st.metric("Total Shots", total_shots)
                            st.metric("On Target", on_target)
                            st.metric("Accuracy", f"{accuracy:.1f}%")
                        
                        with col2:
                            total_xg = sum(s.get('expected_goal', 0) for s in team_shots)
                            xg_per_shot = total_xg / total_shots if total_shots > 0 else 0
                            
                            st.metric("Total xG", f"{total_xg:.2f}")
                            st.metric("xG per Shot", f"{xg_per_shot:.3f}")
                            st.metric("Shots Under Pressure", self.shots_under_pressure[shot_team_filter])
                        
                        with col3:
                            # Shot type distribution
                            shot_types = {}
                            for shot in team_shots:
                                shot_type = shot.get('shot_type', 'normal')
                                shot_types[shot_type] = shot_types.get(shot_type, 0) + 1
                            
                            # Display as horizontal bar chart
                            if shot_types:
                                fig = go.Figure()
                                
                                categories = list(shot_types.keys())
                                values = list(shot_types.values())
                                
                                fig.add_trace(go.Bar(
                                    y=categories,
                                    x=values,
                                    orientation='h',
                                    marker_color=team_color
                                ))
                                
                                fig.update_layout(
                                    height=200,
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    title="Shot Types"
                                )
                                
                                st.plotly_chart(fig)
                        
                        # Shot zones analysis
                        st.subheader("🎯 Shot Zones Analysis")
                        
                        shot_zones = {}
                        for shot in team_shots:
                            zone = shot.get('zone', 'unknown')
                            shot_zones[zone] = shot_zones.get(zone, 0) + 1
                        
                        if shot_zones:
                            # Create a treemap for shot zones
                            fig = px.treemap(
                                names=list(shot_zones.keys()),
                                parents=["" for _ in shot_zones],
                                values=list(shot_zones.values()),
                                title=f"{shot_team_filter} Shots by Zone",
                                color_discrete_sequence=[team_color]
                            )
                            
                            st.plotly_chart(fig)
                        else:
                            st.info("Not enough shot zone data available.")
                    else:
                        st.info(f"No shot data available for {shot_team_filter}.")
                else:
                    st.info("No shot data available. Run the analysis first.")
        except Exception as e:
            st.error(f"Error displaying spatial analysis: {str(e)}")
            traceback.print_exc()
    
    def generate_team_heatmap(self, team):
        """Generate and display heatmap for a specific team"""
        try:
            # Collect all positions for this team
            positions = []
            for player_id, team_name in self.player_team.items():
                if team_name == team and player_id in self.player_positions:
                    positions.extend(self.player_positions[player_id])
            
            if positions:
                # Convert to numpy array
                positions = np.array(positions)
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # Draw field background
                rect = plt.Rectangle((0, 0), self.video_info.target_width, self.video_info.target_height, 
                                    facecolor='#238823', alpha=0.3, edgecolor='white')
                ax.add_patch(rect)
                
                # Draw center line and circle
                ax.plot([self.video_info.target_width/2, self.video_info.target_width/2], [0, self.video_info.target_height], 'white')
                center_circle = plt.Circle((self.video_info.target_width/2, self.video_info.target_height/2), 
                                         self.video_info.target_height/10, fill=False, color='white')
                ax.add_patch(center_circle)
                
                # Generate KDE for heatmap
                x = positions[:, 0]
                y = positions[:, 1]
                
                # Create 2D histogram
                heatmap, xedges, yedges = np.histogram2d(x, y, bins=40, 
                                                      range=[[0, self.video_info.target_width], [0, self.video_info.target_height]])
                
                # Smooth the heatmap
                heatmap = gaussian_filter(heatmap, sigma=1.5)
                
                # Plot heatmap
                c = ax.imshow(heatmap.T, cmap='hot', origin='lower', 
                             extent=[0, self.video_info.target_width, 0, self.video_info.target_height],
                             alpha=0.7, interpolation='bilinear')
                
                # Remove axes
                ax.axis('off')
                
                # Add colorbar
                fig.colorbar(c, ax=ax, label='Presence Intensity')
                
                # Display
                st.pyplot(fig)
            else:
                st.info("Not enough data for heatmap generation.")
        except Exception as e:
            st.error(f"Error generating heatmap: {str(e)}")
            traceback.print_exc()
    
    def display_team_analysis(self):
        """Display team level analysis and comparisons"""
        try:
            with self.tab4:
                st.subheader("📊 Team Analysis")
                
                # Check if team stats are available
                if not self.team_stats:
                    st.info("No team statistics available. Run the analysis first.")
                    return
                
                # Create team comparison dataframe
                metrics = ['Possession (%)', 'Distance (m)', 'Passes', 'Pass Completion (%)', 
                          'Forward Passes (%)', 'Progressive Passes', 'Breaking Lines Passes',
                          'Expected Assists (xA)', 'Shots', 'Shots on Target', 'Shot Accuracy (%)',
                          'Expected Goals (xG)', 'Most Used Formation']
                
                # Make sure all metrics exist in team_stats
                available_metrics = []
                for m in metrics:
                    if m in self.team_stats[self.team_home] and m in self.team_stats[self.team_away]:
                        available_metrics.append(m)
                
                if available_metrics:
                    team_df = pd.DataFrame({
                        'Metric': available_metrics,
                        self.team_home: [self.team_stats[self.team_home].get(m, 'N/A') for m in available_metrics],
                        self.team_away: [self.team_stats[self.team_away].get(m, 'N/A') for m in available_metrics]
                    })
                    
                    # Display team comparison
                    st.subheader("⚔️ Team Comparison")
                    st.dataframe(team_df.set_index('Metric'))
                    
                    # Radar chart for team comparison
                    st.subheader("📊 Team Performance Radar")
                    
                    # Prepare data for radar chart (only use numerical metrics)
                    numerical_metrics = [m for m in available_metrics if m != 'Most Used Formation']
                    
                    if numerical_metrics:
                        # Get values for each category with safety checks
                        home_values = []
                        away_values = []
                        
                        for metric in numerical_metrics:
                            home_value = self.team_stats[self.team_home].get(metric, 0)
                            away_value = self.team_stats[self.team_away].get(metric, 0)
                            
                            # Ensure we have values
                            if isinstance(home_value, (int, float)) and isinstance(away_value, (int, float)):
                                home_values.append(home_value)
                                away_values.append(away_value)
                            else:
                                # Skip this metric
                                continue
                        
                        if home_values and away_values:
                            # Calculate max values for normalization
                            max_values = [max(home_values[i], away_values[i], 1) for i in range(len(home_values))]
                            
                            # Normalize values to 0-1 scale
                            home_values_norm = [home_values[i] / max_values[i] for i in range(len(home_values))]
                            away_values_norm = [away_values[i] / max_values[i] for i in range(len(away_values))]
                            
                            # Plot radar chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatterpolar(
                                r=home_values_norm + [home_values_norm[0]],  # Close the loop
                                theta=numerical_metrics + [numerical_metrics[0]],  # Close the loop
                                fill='toself',
                                name=self.team_home,
                                line_color=self.home_color
                            ))
                            
                            fig.add_trace(go.Scatterpolar(
                                r=away_values_norm + [away_values_norm[0]],  # Close the loop
                                theta=numerical_metrics + [numerical_metrics[0]],  # Close the loop
                                fill='toself',
                                name=self.team_away,
                                line_color=self.away_color
                            ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )),
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig)
                else:
                    st.warning("Team statistics are incomplete or not available.")
                
                # Passing effectiveness comparison
                st.subheader("🔄 Passing Effectiveness")
                
                if self.pass_data:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pass success rate
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=[self.team_home, self.team_away],
                            y=[self.pass_success_rate[self.team_home], self.pass_success_rate[self.team_away]],
                            marker_color=[self.home_color, self.away_color]
                        ))
                        
                        fig.update_layout(
                            title="Pass Completion Rate (%)",
                            yaxis_range=[0, 100]
                        )
                        
                        st.plotly_chart(fig)
                    
                    with col2:
                        # Pass types comparison
                        home_pass_types = dict(self.pass_types[self.team_home])
                        away_pass_types = dict(self.pass_types[self.team_away])
                        
                        # Get all pass types
                        all_types = set(home_pass_types.keys()).union(set(away_pass_types.keys()))
                        
                        # Create data for grouped bar chart
                        pass_type_data = []
                        for pass_type in all_types:
                            pass_type_data.append({
                                'Pass Type': pass_type,
                                self.team_home: home_pass_types.get(pass_type, 0),
                                self.team_away: away_pass_types.get(pass_type, 0)
                            })
                        
                        # Create dataframe for plotting
                        pass_type_df = pd.DataFrame(pass_type_data)
                        
                        # Create grouped bar chart
                        fig = px.bar(
                            pass_type_df,
                            x='Pass Type',
                            y=[self.team_home, self.team_away],
                            barmode='group',
                            title="Pass Types Comparison",
                            color_discrete_map={self.team_home: self.home_color, self.team_away: self.away_color}
                        )
                        
                        st.plotly_chart(fig)
                    
                    # Advanced passing metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Progressive passes comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=[self.team_home, self.team_away],
                            y=[self.progressive_passes[self.team_home], self.progressive_passes[self.team_away]],
                            marker_color=[self.home_color, self.away_color]
                        ))
                        
                        fig.update_layout(title="Progressive Passes")
                        st.plotly_chart(fig)
                    
                    with col2:
                        # Breaking lines passes comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=[self.team_home, self.team_away],
                            y=[self.breaking_lines_passes[self.team_home], self.breaking_lines_passes[self.team_away]],
                            marker_color=[self.home_color, self.away_color]
                        ))
                        
                        fig.update_layout(title="Breaking Lines Passes")
                        st.plotly_chart(fig)
                    
                    with col3:
                        # Expected assists comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=[self.team_home, self.team_away],
                            y=[self.total_xA[self.team_home], self.total_xA[self.team_away]],
                            marker_color=[self.home_color, self.away_color]
                        ))
                        
                        fig.update_layout(title="Expected Assists (xA)")
                        st.plotly_chart(fig)
                else:
                    st.info("No pass data available for comparison.")
                
                # Shot effectiveness comparison
                st.subheader("🎯 Shot Effectiveness")
                
                if self.shot_data:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Shot accuracy comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=[self.team_home, self.team_away],
                            y=[self.shot_success_rate[self.team_home], self.shot_success_rate[self.team_away]],
                            marker_color=[self.home_color, self.away_color]
                        ))
                        
                        fig.update_layout(
                            title="Shot Accuracy (%)",
                            yaxis_range=[0, 100]
                        )
                        
                        st.plotly_chart(fig)
                    
                    with col2:
                        # Expected goals comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=[self.team_home, self.team_away],
                            y=[self.total_xG[self.team_home], self.total_xG[self.team_away]],
                            marker_color=[self.home_color, self.away_color]
                        ))
                        
                        fig.update_layout(title="Expected Goals (xG)")
                        st.plotly_chart(fig)
                else:
                    st.info("No shot data available for comparison.")
                
                # Display most used formations
                st.subheader("🧩 Formation Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### {self.team_home} Formations")
                    if self.team_formations[self.team_home]:
                        formations = dict(sorted(self.team_formations[self.team_home].items(), 
                                               key=lambda x: x[1], reverse=True))
                        
                        # Convert to percentages
                        total = sum(formations.values())
                        if total > 0:  # Avoid division by zero
                            formation_percentages = {k: v/total*100 for k, v in formations.items()}
                            
                            # Create pie chart
                            fig, ax = plt.subplots()
                            ax.pie(formation_percentages.values(), labels=formation_percentages.keys(), 
                                  autopct='%1.1f%%', startangle=90)
                            ax.axis('equal')
                            plt.title(f"Formation Distribution - {self.team_home}")
                            st.pyplot(fig)
                        else:
                            st.info("Insufficient data to determine formations.")
                    else:
                        st.info("No formation data available.")
                
                with col2:
                    st.markdown(f"#### {self.team_away} Formations")
                    if self.team_formations[self.team_away]:
                        formations = dict(sorted(self.team_formations[self.team_away].items(), 
                                               key=lambda x: x[1], reverse=True))
                        
                        # Convert to percentages
                        total = sum(formations.values())
                        if total > 0:  # Avoid division by zero
                            formation_percentages = {k: v/total*100 for k, v in formations.items()}
                            
                            # Create pie chart
                            fig, ax = plt.subplots()
                            ax.pie(formation_percentages.values(), labels=formation_percentages.keys(), 
                                  autopct='%1.1f%%', startangle=90)
                            ax.axis('equal')
                            plt.title(f"Formation Distribution - {self.team_away}")
                            st.pyplot(fig)
                        else:
                            st.info("Insufficient data to determine formations.")
                    else:
                        st.info("No formation data available.")
        except Exception as e:
            st.error(f"Error displaying team analysis: {str(e)}")
            traceback.print_exc()
    
    def analyze_strengths_weaknesses(self):
        """Analyze team strengths and weaknesses based on collected data"""
        try:
            for team in [self.team_home, self.team_away]:
                other_team = self.team_away if team == self.team_home else self.team_home
                
                # Get team metrics
                possession_pct = self.team_stats[team]['Possession (%)']
                total_passes = max(sum(1 for p in self.pass_data if p['team'] == team), 1)  # Avoid division by zero
                completed_passes = sum(1 for p in self.pass_data if p['team'] == team and p.get('completed', True))
                pass_completion = (completed_passes / total_passes * 100) if total_passes > 0 else 0
                
                # Get pass metrics
                pass_completion = self.pass_success_rate[team]
                forward_pass_pct = self.team_stats[team]['Forward Passes (%)']
                progressive_passes = self.progressive_passes[team]
                danger_zone_passes = self.danger_zone_passes[team]
                breaking_lines_passes = self.breaking_lines_passes[team]
                xA = self.total_xA[team]
                
                # Get shot metrics
                total_shots = sum(1 for s in self.shot_data if s['team'] == team)
                shots_on_target = sum(1 for s in self.shot_data if s['team'] == team and s.get('on_target', False))
                shot_accuracy = (shots_on_target / total_shots * 100) if total_shots > 0 else 0
                shots_under_pressure = self.shots_under_pressure[team]
                total_xG = self.total_xG[team]
                
                # Get defensive metrics
                defensive_actions_count = sum(len(actions) for player_id, actions in self.defensive_actions.items() 
                                          if player_id in self.player_team and self.player_team[player_id] == team)
                
                # Team strengths
                strengths = []
                
                # Possession strengths
                if possession_pct > 55:
                    strengths.append(("high_possession_percentage", possession_pct))
                
                # Passing strengths
                if pass_completion > 80:
                    strengths.append(("high_pass_completion", pass_completion))
                
                if forward_pass_pct > 40:
                    strengths.append(("high_forward_passes", forward_pass_pct))
                
                if progressive_passes > 20:
                    strengths.append(("high_progressive_passes", progressive_passes))
                
                if breaking_lines_passes > 15:
                    strengths.append(("high_through_balls", breaking_lines_passes))
                
                if danger_zone_passes > 10:
                    strengths.append(("high_passes_final_third", danger_zone_passes))
                
                if xA > 0.8:
                    strengths.append(("high_xA", xA*100))  # Scale up for visual display
                
                # Shooting strengths
                if shot_accuracy > 40:
                    strengths.append(("high_shots_on_target", shot_accuracy))
                
                if total_xG > 1.0:
                    strengths.append(("high_xG_per_shot", total_xG*100))  # Scale up for visual display
                
                # Defensive strengths 
                if defensive_actions_count > 50:
                    strengths.append(("high_defensive_duels_won", defensive_actions_count))
                
                # Add pressing intensity if available
                pressing_success = max(self.pressing_intensity.get(team, 0), 1)
                if pressing_success > 70:
                    strengths.append(("high_pressing_success", pressing_success))
                
                # If no strengths detected, add some default ones based on best metrics
                if not strengths:
                    # Add possession as a strength (even if it's not high)
                    strengths.append(("high_possession_percentage", max(possession_pct, 1)))
                    # Add pass completion
                    strengths.append(("high_pass_completion", max(pass_completion, 1)))
                    # Add defensive actions
                    strengths.append(("high_defensive_duels_won", max(defensive_actions_count, 1)))
                
                # Sort strengths by value and take top 3
                strengths.sort(key=lambda x: x[1], reverse=True)
                self.team_strengths[team] = {
                    key: {
                        "value": value,
                        "description": self.tactical_strengths.get(key, "Team strength")
                    } for key, value in strengths[:3]
                }
                
                # Team weaknesses
                weaknesses = []
                
                # Possession weaknesses
                if possession_pct < 45:
                    weaknesses.append(("low_possession_percentage", max(possession_pct, 1)))
                
                # Passing weaknesses
                if pass_completion < 70:
                    weaknesses.append(("low_pass_completion", max(pass_completion, 1)))
                
                if forward_pass_pct < 30:
                    weaknesses.append(("low_forward_passes", max(forward_pass_pct, 1)))
                
                if progressive_passes < 10:
                    weaknesses.append(("low_progressive_passes", max(progressive_passes, 1)))
                
                # Shooting weaknesses
                if shot_accuracy < 30:
                    weaknesses.append(("low_shots_on_target", max(shot_accuracy, 1)))
                
                if total_xG < 0.5:
                    weaknesses.append(("low_xG_per_shot", max(total_xG*100, 1)))  # Scale up for display
                
                # Defensive weaknesses
                if defensive_actions_count < 30:
                    weaknesses.append(("low_defensive_duels_won", max(defensive_actions_count, 1)))
                
                if pressing_success < 50:
                    weaknesses.append(("low_pressing_success", max(pressing_success, 1)))
                
                # Calculate lateral passes
                lateral_passes = sum(1 for p in self.pass_data if p['team'] == team and p.get('direction') == 'lateral')
                lateral_pass_pct = (lateral_passes / total_passes * 100) if total_passes > 0 else 0
                
                if lateral_pass_pct > 40:
                    weaknesses.append(("high_lateral_passes", lateral_pass_pct))
                
                # Check if team is vulnerable during defensive transitions
                if shots_under_pressure > 5:
                    weaknesses.append(("poor_defensive_transitions", shots_under_pressure))
                
                # If no weaknesses detected, add some default ones
                if not weaknesses:
                    # Choose metrics that are relatively lower compared to others
                    if possession_pct < pass_completion:
                        weaknesses.append(("low_possession_percentage", max(possession_pct, 1)))
                    else:
                        weaknesses.append(("low_pass_completion", max(pass_completion, 1)))
                    
                    if shot_accuracy < 50:
                        weaknesses.append(("low_shots_on_target", max(shot_accuracy, 1)))
                    
                    weaknesses.append(("low_pressing_success", max(pressing_success, 1)))
                
                # Sort weaknesses by value (lower is worse)
                weaknesses.sort(key=lambda x: x[1])
                self.team_weaknesses[team] = {
                    key: {
                        "value": value,
                        "description": self.tactical_weaknesses.get(key, "Area for improvement")
                    } for key, value in weaknesses[:3]
                }
                
                # Generate tactical suggestions based on opponent's weaknesses
                self.generate_tactical_suggestions(team, other_team)
        except Exception as e:
            st.error(f"Error analyzing strengths and weaknesses: {str(e)}")
            traceback.print_exc()
    
    def generate_tactical_suggestions(self, team, opponent):
        """Generate tactical suggestions based on opponent weaknesses and playstyle"""
        try:
            suggestions = []
            
            # Get opponent's playstyle
            opponent_style = self.away_playstyle if opponent == self.team_away else self.home_playstyle
            
            # If using Gemini and insights are available, prioritize those
            if self.enable_gemini and self.gemini_insights[team]:
                # Filter for only suggestion content
                suggestions = [insight for insight in self.gemini_insights[team] 
                              if "suggestion" in insight.lower() or "recommendation" in insight.lower()]
                
                # If we found suggestions, use them
                if suggestions:
                    # Clean up the suggestions
                    cleaned_suggestions = []
                    for suggestion in suggestions:
                        # Split by lines and remove empty ones
                        lines = [line.strip() for line in suggestion.split('\n') if line.strip()]
                        cleaned_suggestions.extend(lines)
                    
                    # Keep only items that look like suggestions (not headers)
                    real_suggestions = [s for s in cleaned_suggestions 
                                       if not s.startswith('#') and not s.endswith(':') 
                                       and len(s) > 15]
                    
                    # If we have enough suggestions from Gemini, use them
                    if len(real_suggestions) >= 3:
                        self.tactical_suggestions[team] = real_suggestions[:5]
                        return
            
            # If we don't have Gemini suggestions or not enough, fall back to our built-in logic
            
            # Add counter-strategy suggestions based on opponent's playstyle
            if opponent_style in self.counter_strategies:
                suggestions.extend(self.counter_strategies[opponent_style])
            else:
                # Default suggestions if playstyle not recognized
                suggestions.extend(self.counter_strategies["Custom"])
            
            # Add specific suggestions based on opponent's weaknesses
            for weakness_key in self.team_weaknesses.get(opponent, {}):
                if weakness_key == "low_possession_percentage":
                    suggestions.append("Apply high pressing to force turnovers")
                elif weakness_key == "low_pass_completion":
                    suggestions.append("Press aggressively during build-up phase")
                elif weakness_key == "low_shots_on_target":
                    suggestions.append("Allow low-quality shots from distance while protecting the box")
                elif weakness_key == "low_pressing_success":
                    suggestions.append("Use technical midfielders to bypass their press")
                elif weakness_key == "low_defensive_duels_won":
                    suggestions.append("Target 1v1 situations in attack")
                elif weakness_key == "low_forward_passes":
                    suggestions.append("Block forward passing lanes to force backward/sideways passes")
                elif weakness_key == "high_lateral_passes":
                    suggestions.append("Set pressing traps on sidelines to win ball during sideways passes")
                elif weakness_key == "low_progressive_passes":
                    suggestions.append("Press aggressively in midfield to prevent ball progression")
                elif weakness_key == "poor_defensive_transitions":
                    suggestions.append("Counter-attack quickly after winning possession")
                elif weakness_key == "low_xG_per_shot":
                    suggestions.append("Compact defense to force low-quality shots from distance")
            
            # Add suggestions based on team's strengths
            for strength_key in self.team_strengths.get(team, {}):
                if strength_key == "high_possession_percentage":
                    suggestions.append("Focus on patient build-up play to capitalize on possession advantage")
                elif strength_key == "high_pass_completion":
                    suggestions.append("Use quick passing combinations to break through defensive lines")
                elif strength_key == "high_shots_on_target":
                    suggestions.append("Create more shooting opportunities for forwards")
                elif strength_key == "high_pressing_success":
                    suggestions.append("Implement aggressive pressing triggers in opponent's half")
                elif strength_key == "high_defensive_duels_won":
                    suggestions.append("Encourage defenders to step up for interceptions")
                elif strength_key == "high_forward_passes":
                    suggestions.append("Maintain vertical passing options to exploit progressive passing")
                elif strength_key == "high_through_balls":
                    suggestions.append("Position forwards to make runs for through passes behind defense")
                elif strength_key == "high_progressive_passes":
                    suggestions.append("Create space in midfield to continue progressive passing patterns")
                elif strength_key == "high_passes_final_third":
                    suggestions.append("Position additional players in the final third to capitalize on chances")
                elif strength_key == "high_xA" or strength_key == "high_xG_per_shot":
                    suggestions.append("Prioritize getting the ball to creative players in dangerous areas")
            
            # Ensure we have at least 5 suggestions
            if len(suggestions) < 5:
                default_suggestions = [
                    "Focus on maintaining defensive organization",
                    "Exploit spaces in wide areas",
                    "Increase tempo in transition phases",
                    "Use defensive midfielder to screen opponent attacks",
                    "Maintain compact shape between defensive lines"
                ]
                # Add default suggestions until we have at least 5
                for suggestion in default_suggestions:
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)
                    if len(suggestions) >= 5:
                        break
            
            # Store unique suggestions (up to 5)
            self.tactical_suggestions[team] = list(set(suggestions))[:5]
        except Exception as e:
            st.error(f"Error generating tactical suggestions: {str(e)}")
            traceback.print_exc()
            # Set default suggestions in case of error
            self.tactical_suggestions[team] = [
                "Maintain defensive organization",
                "Focus on possession in the middle third",
                "Press when the ball is in wide areas",
                "Create numerical advantages in attack",
                "Stay compact between the lines"
            ]
    
    def display_strengths_weaknesses(self):
        """Display team strengths and weaknesses"""
        try:
            with self.tab5:
                st.subheader("💪 Team Strengths & Weaknesses Analysis")
                
                # Check if strengths and weaknesses are available
                if not self.team_strengths[self.team_home] and not self.team_strengths[self.team_away]:
                    st.info("Strengths and weaknesses analysis not available. Run the analysis first.")
                    return
                
                col1, col2 = st.columns(2)
                
                # Home team analysis
                with col1:
                    st.markdown(f"### {self.team_home}")
                    
                    # Strengths
                    st.markdown("#### 💪 Strengths")
                    if self.team_strengths[self.team_home]:
                        for key, data in self.team_strengths[self.team_home].items():
                            st.markdown(f"**{data['description']}** _{key}_")
                            # Create progress bar
                            progress_value = min(max(data['value'], 0.1) / 100, 1.0)  # Ensure non-zero
                            st.progress(progress_value)
                    else:
                        st.info("No significant strengths identified.")
                    
                    # Weaknesses
                    st.markdown("#### 🔍 Areas for Improvement")
                    if self.team_weaknesses[self.team_home]:
                        for key, data in self.team_weaknesses[self.team_home].items():
                            st.markdown(f"**{data['description']}** _{key}_")
                            # Create inverted progress bar for weaknesses
                            progress_value = min(max(data['value'], 0.1) / 100, 1.0)  # Ensure non-zero
                            st.progress(progress_value)
                    else:
                        st.info("No significant weaknesses identified.")
                
                # Away team analysis
                with col2:
                    st.markdown(f"### {self.team_away}")
                    
                    # Strengths
                    st.markdown("#### 💪 Strengths")
                    if self.team_strengths[self.team_away]:
                        for key, data in self.team_strengths[self.team_away].items():
                            st.markdown(f"**{data['description']}** _{key}_")
                            # Create progress bar
                            progress_value = min(max(data['value'], 0.1) / 100, 1.0)  # Ensure non-zero
                            st.progress(progress_value)
                    else:
                        st.info("No significant strengths identified.")
                    
                    # Weaknesses
                    st.markdown("#### 🔍 Areas for Improvement")
                    if self.team_weaknesses[self.team_away]:
                        for key, data in self.team_weaknesses[self.team_away].items():
                            st.markdown(f"**{data['description']}** _{key}_")
                            # Create inverted progress bar for weaknesses
                            progress_value = min(max(data['value'], 0.1) / 100, 1.0)  # Ensure non-zero
                            st.progress(progress_value)
                    else:
                        st.info("No significant weaknesses identified.")
                
                # AI Insights section
                if self.enable_gemini and self.gemini_insights[self.team_home]:
                    st.subheader("🤖 AI Tactical Insights")
                    
                    # Display AI insights in an expandable section
                    with st.expander("View AI Tactical Analysis", expanded=True):
                        for insight in self.gemini_insights[self.team_home]:
                            if "key tactical observations" in insight.lower() or "observation" in insight.lower():
                                st.markdown(insight)
                
                # Visualize strengths and weaknesses as radar charts
                st.subheader("📊 Strengths & Weaknesses Comparison")
                
                # Create combined radar chart for both teams
                if self.team_strengths[self.team_home] and self.team_strengths[self.team_away]:
                    # Get all unique strength categories
                    all_strengths = set(self.team_strengths[self.team_home].keys()).union(
                        set(self.team_strengths[self.team_away].keys())
                    )
                    
                    # Create normalized values for radar chart
                    home_strength_values = []
                    away_strength_values = []
                    strength_categories = []
                    
                    for strength in all_strengths:
                        strength_categories.append(self.tactical_strengths.get(strength, strength))
                        
                        # Get values with default of 0 if not present
                        home_val = self.team_strengths[self.team_home].get(strength, {"value": 0})["value"]
                        away_val = self.team_strengths[self.team_away].get(strength, {"value": 0})["value"]
                        
                        # Normalize to 0-1 scale
                        max_val = max(home_val, away_val, 1)
                        home_strength_values.append(home_val / max_val)
                        away_strength_values.append(away_val / max_val)
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=home_strength_values + [home_strength_values[0]],  # Close the loop
                        theta=strength_categories + [strength_categories[0]],  # Close the loop
                        fill='toself',
                        name=f"{self.team_home} Strengths",
                        line_color=self.home_color
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=away_strength_values + [away_strength_values[0]],  # Close the loop
                        theta=strength_categories + [strength_categories[0]],  # Close the loop
                        fill='toself',
                        name=f"{self.team_away} Strengths",
                        line_color=self.away_color
                    ))
                    
                    fig.update_layout(
                        title="Team Strengths Comparison",
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig)
                
                # SWOT Analysis for teams
                st.subheader("📈 SWOT Analysis")
                
                # Create SWOT analysis table for both teams
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### {self.team_home} SWOT Analysis")
                    
                    strengths_list = [f"- {data['description']}" for key, data in self.team_strengths[self.team_home].items()]
                    weaknesses_list = [f"- {data['description']}" for key, data in self.team_weaknesses[self.team_home].items()]
                    
                    # Generate opportunities based on opponent weaknesses
                    opportunities = []
                    for key, data in self.team_weaknesses[self.team_away].items():
                        if key == "low_possession_percentage":
                            opportunities.append("- Dominate possession to control game flow")
                        elif key == "low_pass_completion":
                            opportunities.append("- Apply pressure to force passing errors")
                        elif key == "low_shots_on_target":
                            opportunities.append("- Allow long shots while protecting key areas")
                        elif key == "low_pressing_success":
                            opportunities.append("- Build from the back with confidence")
                        elif key == "low_forward_passes":
                            opportunities.append("- Block forward passing lanes to frustrate build-up")
                        elif key == "high_lateral_passes":
                            opportunities.append("- Set pressing traps on sidelines")
                    
                    # Generate threats based on opponent strengths
                    threats = []
                    for key, data in self.team_strengths[self.team_away].items():
                        if key == "high_possession_percentage":
                            threats.append("- May struggle to gain possession")
                        elif key == "high_pass_completion":
                            threats.append("- Opponent's efficient passing may break down defense")
                        elif key == "high_shots_on_target":
                            threats.append("- Vulnerable to quality finishing")
                        elif key == "high_pressing_success":
                            threats.append("- May struggle to build up from back under pressure")
                        elif key == "high_forward_passes":
                            threats.append("- Vulnerable to quick vertical progression")
                        elif key == "high_through_balls":
                            threats.append("- Vulnerable to penetrative passes behind defense")
                    
                    # Ensure we have at least some items in each category
                    if not opportunities:
                        opportunities = ["- Force turnovers in opponent's half", "- Exploit wide areas in attack"]
                    if not threats:
                        threats = ["- Opponent's counter-attacks", "- Set pieces against"]
                    
                    # Create SWOT markdown table
                    swot_table = """
                    | Strengths | Weaknesses |
                    | --- | --- |
                    | {} | {} |
                    
                    | Opportunities | Threats |
                    | --- | --- |
                    | {} | {} |
                    """.format(
                        "<br>".join(strengths_list) or "- No significant strengths identified",
                        "<br>".join(weaknesses_list) or "- No significant weaknesses identified",
                        "<br>".join(opportunities[:3]),
                        "<br>".join(threats[:3])
                    )
                    
                    st.markdown(swot_table, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"#### {self.team_away} SWOT Analysis")
                    
                    strengths_list = [f"- {data['description']}" for key, data in self.team_strengths[self.team_away].items()]
                    weaknesses_list = [f"- {data['description']}" for key, data in self.team_weaknesses[self.team_away].items()]
                    
                    # Generate opportunities based on opponent weaknesses
                    opportunities = []
                    for key, data in self.team_weaknesses[self.team_home].items():
                        if key == "low_possession_percentage":
                            opportunities.append("- Dominate possession to control game flow")
                        elif key == "low_pass_completion":
                            opportunities.append("- Apply pressure to force passing errors")
                        elif key == "low_shots_on_target":
                            opportunities.append("- Allow long shots while protecting key areas")
                        elif key == "low_pressing_success":
                            opportunities.append("- Build from the back with confidence")
                        elif key == "low_forward_passes":
                            opportunities.append("- Block forward passing lanes to frustrate build-up")
                        elif key == "high_lateral_passes":
                            opportunities.append("- Set pressing traps on sidelines")
                    
                    # Generate threats based on opponent strengths
                    threats = []
                    for key, data in self.team_strengths[self.team_home].items():
                        if key == "high_possession_percentage":
                            threats.append("- May struggle to gain possession")
                        elif key == "high_pass_completion":
                            threats.append("- Opponent's efficient passing may break down defense")
                        elif key == "high_shots_on_target":
                            threats.append("- Vulnerable to quality finishing")
                        elif key == "high_pressing_success":
                            threats.append("- May struggle to build up from back under pressure")
                        elif key == "high_forward_passes":
                            threats.append("- Vulnerable to quick vertical progression")
                        elif key == "high_through_balls":
                            threats.append("- Vulnerable to penetrative passes behind defense")
                    
                    # Ensure we have at least some items in each category
                    if not opportunities:
                        opportunities = ["- Force turnovers in opponent's half", "- Exploit wide areas in attack"]
                    if not threats:
                        threats = ["- Opponent's counter-attacks", "- Set pieces against"]
                    
                    # Create SWOT markdown table
                    swot_table = """
                    | Strengths | Weaknesses |
                    | --- | --- |
                    | {} | {} |
                    
                    | Opportunities | Threats |
                    | --- | --- |
                    | {} | {} |
                    """.format(
                        "<br>".join(strengths_list) or "- No significant strengths identified",
                        "<br>".join(weaknesses_list) or "- No significant weaknesses identified",
                        "<br>".join(opportunities[:3]),
                        "<br>".join(threats[:3])
                    )
                    
                    st.markdown(swot_table, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying strengths and weaknesses: {str(e)}")
            traceback.print_exc()
    
    def display_tactical_suggestions(self):
        """Display tactical suggestions based on analysis"""
        try:
            with self.tab6:
                st.subheader("🎯 Tactical Analysis & Suggestions")
                
                # Check if tactical suggestions are available
                if not self.tactical_suggestions[self.team_home] and not self.tactical_suggestions[self.team_away]:
                    st.info("Tactical suggestions not available. Run the analysis with tactical analysis enabled.")
                    return
                
                # Let user select which team to view suggestions for
                selected_team = st.radio(
                    "Select Team for Tactical Analysis",
                    [self.team_home, self.team_away],
                    horizontal=True
                )
                
                opponent = self.team_away if selected_team == self.team_home else self.team_home
                
                # Display opponent analysis
                st.subheader(f"Opponent Analysis: {opponent}")
                
                # Display opponent's playing style
                opponent_style = self.away_playstyle if opponent == self.team_away else self.home_playstyle
                
                # Display style characteristics
                if opponent_style in self.playstyle_patterns:
                    style_data = self.playstyle_patterns[opponent_style]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Playing Style", opponent_style)
                        st.metric("Pass Length", style_data["pass_length"].capitalize())
                    
                    with col2:
                        st.metric("Defensive Line", style_data["defensive_line"].capitalize())
                        st.metric("Pass Tempo", style_data["pass_tempo"].capitalize())
                    
                    with col3:
                        st.metric("Pressing Intensity", style_data["pressing_intensity"].capitalize())
                        st.metric("Width", style_data["width"].capitalize())
                elif opponent_style == "Custom":
                    st.info(f"Custom playing style for {opponent}")
                    custom_attr = f"{opponent.lower().replace(' ', '_')}_playstyle_custom"
                    if hasattr(self, custom_attr):
                        custom_style = getattr(self, custom_attr)
                        st.write(f"Description: {custom_style}")
                
                # Opponent weaknesses display
                st.subheader(f"Key Vulnerabilities of {opponent}")
                
                # Display opponent weaknesses as cards
                if self.team_weaknesses[opponent]:
                    cols = st.columns(len(self.team_weaknesses[opponent]))
                    
                    for i, (key, data) in enumerate(self.team_weaknesses[opponent].items()):
                        with cols[i]:
                            st.markdown(f"""
                            <div style="padding: 10px; border: 1px solid #f0f0f0; border-radius: 5px; margin-bottom: 10px;">
                                <h5 style="color: #ff4b4b;">{data['description']}</h5>
                                <div style="height: 4px; background-color: #ff4b4b; width: {min(max(data['value'], 5), 100)}%;"></div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No significant weaknesses identified for the opponent.")
                
                # Display tactical suggestions
                st.subheader("🎯 Tactical Suggestions")
                
                if self.tactical_suggestions.get(selected_team):
                    for i, suggestion in enumerate(self.tactical_suggestions[selected_team]):
                        # Create a card-like display for each suggestion
                        st.markdown(f"""
                        <div style="padding: 15px; border-left: 5px solid {self.home_color if selected_team == self.team_home else self.away_color}; 
                                background-color: rgba(0,0,0,0.03); margin-bottom: 15px; border-radius: 0 5px 5px 0;">
                            <h4>Strategy {i+1}</h4>
                            <p>{suggestion}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No tactical suggestions available for this team.")
                
                # Game plan visualization
                st.subheader("📋 Game Plan Visualization")
                
                # Use team's most common formation for the game plan
                formation = self.team_stats[selected_team]['Most Used Formation']
                
                if formation != "N/A":
                    # Create a simplified football field visualization
                    fig, ax = plt.subplots(figsize=(10, 7))
                    
                    # Draw field
                    rect = plt.Rectangle((0, 0), 100, 70, 
                                        facecolor='#238823', alpha=0.3, edgecolor='white')
                    ax.add_patch(rect)
                    
                    # Draw field lines
                    ax.plot([50, 50], [0, 70], 'white')  # Center line
                    center_circle = plt.Circle((50, 35), 10, fill=False, color='white')
                    ax.add_patch(center_circle)
                    
                    # Draw penalty areas
                    ax.add_patch(plt.Rectangle((0, 17.5), 16.5, 35, fill=False, edgecolor='white'))
                    ax.add_patch(plt.Rectangle((83.5, 17.5), 16.5, 35, fill=False, edgecolor='white'))
                    
                    # Draw 6-yard boxes
                    ax.add_patch(plt.Rectangle((0, 24.5), 5.5, 21, fill=False, edgecolor='white'))
                    ax.add_patch(plt.Rectangle((94.5, 24.5), 5.5, 21, fill=False, edgecolor='white'))
                    
                    # Get players from formation
                    formation_parts = formation.split('-')
                    
                    # Add players based on formation (simplified)
                    # Goalkeeper
                    ax.scatter(5, 35, s=300, color=self.home_color if selected_team == self.team_home else self.away_color, 
                               edgecolor='white', zorder=3)
                    ax.text(5, 35, "GK", color='white', ha='center', va='center', weight='bold')
                    
                    # Defenders
                    if len(formation_parts) > 0:
                        defenders = int(formation_parts[0])
                        defender_spacing = 60 / (defenders + 1)
                        for i in range(defenders):
                            y_pos = 5 + (i + 1) * defender_spacing
                            ax.scatter(15, y_pos, s=300, color=self.home_color if selected_team == self.team_home else self.away_color, 
                                       edgecolor='white', zorder=3)
                            ax.text(15, y_pos, f"D{i+1}", color='white', ha='center', va='center', weight='bold')
                    
                    # Midfielders
                    if len(formation_parts) > 1:
                        midfielders = int(formation_parts[1])
                        midfielder_spacing = 60 / (midfielders + 1)
                        for i in range(midfielders):
                            y_pos = 5 + (i + 1) * midfielder_spacing
                            ax.scatter(40, y_pos, s=300, color=self.home_color if selected_team == self.team_home else self.away_color, 
                                       edgecolor='white', zorder=3)
                            ax.text(40, y_pos, f"M{i+1}", color='white', ha='center', va='center', weight='bold')
                    
                    # Forwards
                    if len(formation_parts) > 2:
                        forwards = int(formation_parts[2])
                        forward_spacing = 60 / (forwards + 1)
                        for i in range(forwards):
                            y_pos = 5 + (i + 1) * forward_spacing
                            ax.scatter(70, y_pos, s=300, color=self.home_color if selected_team == self.team_home else self.away_color, 
                                       edgecolor='white', zorder=3)
                            ax.text(70, y_pos, f"F{i+1}", color='white', ha='center', va='center', weight='bold')
                    
                    # Add tactical indicators based on suggestions
                    suggestions = self.tactical_suggestions[selected_team]
                    
                    for i, suggestion in enumerate(suggestions[:3]):  # Add visuals for up to 3 suggestions
                        if "press" in suggestion.lower():
                            # Draw pressing indicators in opponent's half
                            for x, y in [(80, 20), (85, 35), (80, 50)]:
                                circle = plt.Circle((x, y), 5, fill=False, color='yellow', linestyle='dashed')
                                ax.add_patch(circle)
                                ax.text(x, y, "P", color='yellow', ha='center', va='center', weight='bold')
                        
                        if "compact" in suggestion.lower() or "defensive" in suggestion.lower():
                            # Draw defensive block
                            rect = plt.Rectangle((25, 15), 30, 40, fill=False, color='blue', linestyle='dashed')
                            ax.add_patch(rect)
                        
                        if "wide" in suggestion.lower():
                            # Draw wide attacking movements
                            ax.arrow(40, 10, 25, -5, head_width=3, head_length=3, fc='cyan', ec='cyan', zorder=4)
                            ax.arrow(40, 60, 25, 5, head_width=3, head_length=3, fc='cyan', ec='cyan', zorder=4)
                        
                        if "counter" in suggestion.lower():
                            # Draw counter attack arrows
                            ax.arrow(20, 35, 60, 0, head_width=3, head_length=3, fc='red', ec='red', zorder=4)
                    
                    # Set plot limits and hide ticks
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 70)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Set title
                    ax.set_title(f"Game Plan for {selected_team} ({formation})")
                    
                    # Show the plot
                    st.pyplot(fig)
                else:
                    st.info("Formation data not available for game plan visualization.")
                
                # AI-Generated tactical insights from Gemini if available
                if self.enable_gemini and selected_team in self.gemini_insights and self.gemini_insights[selected_team]:
                    st.subheader("🤖 AI-Generated Tactical Insight")
                    
                    # Find tactical suggestions from Gemini
                    ai_suggestions = []
                    for insight in self.gemini_insights[selected_team]:
                        if f"suggestions for {selected_team}" in insight.lower():
                            ai_suggestions.append(insight)
                    
                    # Display AI suggestions if available
                    if ai_suggestions:
                        with st.expander("AI Tactical Analysis", expanded=True):
                            for suggestion in ai_suggestions:
                                st.markdown(suggestion)
                    else:
                        # Show general insights
                        with st.expander("AI Match Analysis", expanded=True):
                            for insight in self.gemini_insights[selected_team][:2]:  # Show up to 2 insights
                                st.markdown(insight)
        except Exception as e:
            st.error(f"Error displaying tactical suggestions: {str(e)}")
            traceback.print_exc()
    
    def generate_gemini_insights(self):
        """Generate tactical insights using Gemini API"""
        if not st.session_state.gemini_api_key:
            st.warning("Gemini API key not provided. Skipping AI insights.")
            return
            
        try:
            # Prepare data for Gemini
            team_stats = {
                self.team_home: {
                    'Possession (%)': round(self.team_stats[self.team_home]['Possession (%)'], 2),
                    'Distance (m)': round(self.team_stats[self.team_home]['Distance (m)'], 2),
                    'Passes': self.team_stats[self.team_home]['Passes'],
                    'Pass Completion (%)': self.team_stats[self.team_home].get('Pass Completion (%)', 0),
                    'Progressive Passes': self.team_stats[self.team_home].get('Progressive Passes', 0),
                    'Breaking Lines Passes': self.team_stats[self.team_home].get('Breaking Lines Passes', 0),
                    'Shots': self.team_stats[self.team_home]['Shots'],
                    'Shots on Target': self.team_stats[self.team_home].get('Shots on Target', 0),
                    'Shot Accuracy (%)': self.team_stats[self.team_home].get('Shot Accuracy (%)', 0),
                    'Expected Goals (xG)': self.team_stats[self.team_home].get('Expected Goals (xG)', 0),
                    'Formation': self.team_stats[self.team_home]['Most Used Formation'],
                    'Style': self.home_playstyle
                },
                self.team_away: {
                    'Possession (%)': round(self.team_stats[self.team_away]['Possession (%)'], 2),
                    'Distance (m)': round(self.team_stats[self.team_away]['Distance (m)'], 2),
                    'Passes': self.team_stats[self.team_away]['Passes'],
                    'Pass Completion (%)': self.team_stats[self.team_away].get('Pass Completion (%)', 0),
                    'Progressive Passes': self.team_stats[self.team_away].get('Progressive Passes', 0),
                    'Breaking Lines Passes': self.team_stats[self.team_away].get('Breaking Lines Passes', 0),
                    'Shots': self.team_stats[self.team_away]['Shots'],
                    'Shots on Target': self.team_stats[self.team_away].get('Shots on Target', 0),
                    'Shot Accuracy (%)': self.team_stats[self.team_away].get('Shot Accuracy (%)', 0),
                    'Expected Goals (xG)': self.team_stats[self.team_away].get('Expected Goals (xG)', 0),
                    'Formation': self.team_stats[self.team_away]['Most Used Formation'],
                    'Style': self.away_playstyle
                }
            }
            
            # Prepare pass data
            pass_stats = {
                self.team_home: {
                    'total': sum(1 for p in self.pass_data if p['team'] == self.team_home),
                    'forward': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('direction') == 'forward'),
                    'backward': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('direction') == 'backward'),
                    'lateral': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('direction') == 'lateral'),
                    'progressive': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('progressive', False)),
                    'breaking_lines': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('breaking_lines', False)),
                    'danger_zone': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('danger_zone', False)),
                    'types': dict(self.pass_types[self.team_home])
                },
                self.team_away: {
                    'total': sum(1 for p in self.pass_data if p['team'] == self.team_away),
                    'forward': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('direction') == 'forward'),
                    'backward': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('direction') == 'backward'),
                    'lateral': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('direction') == 'lateral'),
                    'progressive': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('progressive', False)),
                    'breaking_lines': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('breaking_lines', False)),
                    'danger_zone': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('danger_zone', False)),
                    'types': dict(self.pass_types[self.team_away])
                }
            }
            
            # Prepare shot data
            shot_stats = {
                self.team_home: {
                    'total': sum(1 for s in self.shot_data if s['team'] == self.team_home),
                    'on_target': sum(1 for s in self.shot_data if s['team'] == self.team_home and s.get('on_target', False)),
                    'xG': round(sum(s.get('expected_goal', 0) for s in self.shot_data if s['team'] == self.team_home), 3),
                    'under_pressure': self.shots_under_pressure[self.team_home],
                    'zones': dict(self.shot_zones[self.team_home]),
                    'types': dict(self.shot_types[self.team_home])
                },
                self.team_away: {
                    'total': sum(1 for s in self.shot_data if s['team'] == self.team_away),
                    'on_target': sum(1 for s in self.shot_data if s['team'] == self.team_away and s.get('on_target', False)),
                    'xG': round(sum(s.get('expected_goal', 0) for s in self.shot_data if s['team'] == self.team_away), 3),
                    'under_pressure': self.shots_under_pressure[self.team_away],
                    'zones': dict(self.shot_zones[self.team_away]),
                    'types': dict(self.shot_types[self.team_away])
                }
            }
            
            # Create a comprehensive prompt for Gemini
            prompt = f"""
            You are a professional football analyst with deep expertise in tactical analysis. 
            Analyze the following match data between {self.team_home} vs {self.team_away} and provide detailed tactical insights:
            
            # Match Information
            - Teams: {self.team_home} vs {self.team_away}
            - Home team playing style: {self.home_playstyle}
            - Away team playing style: {self.away_playstyle}
            
            # Team Stats
            {json.dumps(team_stats, indent=2)}
            
            # Pass Stats
            {json.dumps(pass_stats, indent=2)}
            
            # Shot Stats
            {json.dumps(shot_stats, indent=2)}
            
            # REQUIRED ANALYSIS
            Please provide ALL of the following sections:
            
            ## 1. Key Tactical Observations (5 points)
            Analyze the overall tactical approach of both teams based on the statistics. Focus on patterns, strategies, and key performance indicators.
            
            ## 2. Team Strengths - {self.team_home} (3 key points)
            Identify three specific strengths of {self.team_home} based on the data.
            
            ## 3. Team Weaknesses - {self.team_home} (3 key points)
            Identify three areas of improvement for {self.team_home} based on the data.
            
            ## 4. Team Strengths - {self.team_away} (3 key points)
            Identify three specific strengths of {self.team_away} based on the data.
            
            ## 5. Team Weaknesses - {self.team_away} (3 key points)
            Identify three areas of improvement for {self.team_away} based on the data.
            
            ## 6. Tactical Suggestions for {self.team_home} (5 specific points)
            Provide 5 detailed, specific tactical recommendations for {self.team_home} to exploit {self.team_away}'s weaknesses and counter their strengths.
            
            ## 7. Tactical Suggestions for {self.team_away} (5 specific points)
            Provide 5 detailed, specific tactical recommendations for {self.team_away} to exploit {self.team_home}'s weaknesses and counter their strengths.
            
            # Guidelines
            - Focus on tactical aspects, not just statistics
            - Provide specific, concrete observations and recommendations
            - Use football terminology and concepts
            - Explain your reasoning for each observation and suggestion
            - Be specific and detailed in your suggestions
            """
            
            # Make API call to Gemini
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 4096
                }
            }
            
            with st.spinner("Generating AI tactical insights..."):
                response = requests.post(
                    f"{api_url}?key={st.session_state.gemini_api_key}",
                    headers=headers,
                    json=data
                )
                
                # Process response
                if response.status_code == 200:
                    response_json = response.json()
                    
                    if 'candidates' in response_json and len(response_json['candidates']) > 0:
                        text_response = response_json['candidates'][0]['content']['parts'][0]['text']
                        
                        # Parse the response into different sections
                        sections = text_response.split("##")
                        
                        # Store insights
                        for section in sections:
                            # Skip empty sections
                            if not section.strip():
                                continue
                                
                            # Store observation in both teams' insights
                            if "Key Tactical Observations" in section:
                                self.gemini_insights[self.team_home].append(f"## {section.strip()}")
                                self.gemini_insights[self.team_away].append(f"## {section.strip()}")
                            # Store team-specific strengths and weaknesses
                            elif f"Team Strengths - {self.team_home}" in section:
                                self.gemini_insights[self.team_home].append(f"## {section.strip()}")
                            elif f"Team Weaknesses - {self.team_home}" in section:
                                self.gemini_insights[self.team_home].append(f"## {section.strip()}")
                            elif f"Team Strengths - {self.team_away}" in section:
                                self.gemini_insights[self.team_away].append(f"## {section.strip()}")
                            elif f"Team Weaknesses - {self.team_away}" in section:
                                self.gemini_insights[self.team_away].append(f"## {section.strip()}")
                            # Store team-specific tactical suggestions
                            elif f"Tactical Suggestions for {self.team_home}" in section:
                                self.gemini_insights[self.team_home].append(f"## {section.strip()}")
                                # Extract suggestions for tactical suggestions
                                lines = section.strip().split("\n")[1:]  # Skip the heading
                                suggestions = []
                                for line in lines:
                                    line = line.strip()
                                    if line and (line[0].isdigit() or line[0] == '-'):
                                        suggestions.append(line[2:] if line[0] in ('1', '2', '3', '4', '5', '-') else line)
                                
                                if suggestions:
                                    self.tactical_suggestions[self.team_home] = suggestions[:5]
                            elif f"Tactical Suggestions for {self.team_away}" in section:
                                self.gemini_insights[self.team_away].append(f"## {section.strip()}")
                                # Extract suggestions for tactical suggestions
                                lines = section.strip().split("\n")[1:]  # Skip the heading
                                suggestions = []
                                for line in lines:
                                    line = line.strip()
                                    if line and (line[0].isdigit() or line[0] == '-'):
                                        suggestions.append(line[2:] if line[0] in ('1', '2', '3', '4', '5', '-') else line)
                                
                                if suggestions:
                                    self.tactical_suggestions[self.team_away] = suggestions[:5]
                        
                        st.success("✅ Generated AI insights successfully!")
                    else:
                        st.warning("Couldn't extract insights from Gemini response.")
                else:
                    st.error(f"Error calling Gemini API: {response.status_code} - {response.text}")
                    
        except Exception as e:
            st.error(f"Error generating AI insights: {str(e)}")
            traceback.print_exc()
    
    def generate_report(self):
        """Generate a comprehensive PDF report with analysis results"""
        try:
            with self.tab7:
                st.subheader("📝 Match Analysis Report")
                
                # Check if analysis data is available
                if not hasattr(self, 'player_stats_df') or self.player_stats_df.empty:
                    st.warning("Analysis data not available. Run the analysis first.")
                    return
                
                # Generate enhanced PDF report
                pdf = FPDF()
                
                # Set up font and margins
                pdf.set_margins(15, 15, 15)
                pdf.set_auto_page_break(auto=True, margin=15)
                
                # Custom function for adding section headers
                def add_section_header(pdf, title):
                    pdf.set_font("Arial", "B", 16)
                    pdf.set_fill_color(230, 230, 230)
                    pdf.cell(180, 10, title, 0, 1, 'L', True)
                    pdf.ln(5)
                
                # ---- Title page ----
                pdf.add_page()
                pdf.set_font("Arial", "B", 24)
                pdf.cell(180, 20, "Football Match Analysis", ln=True, align="C")
                pdf.cell(180, 20, f"{self.team_home} vs {self.team_away}", ln=True, align="C")
                
                # Add date and time
                pdf.set_font("Arial", "", 12)
                pdf.cell(180, 10, f"Analysis Date: {time.strftime('%Y-%m-%d')}", ln=True, align="C")
                pdf.cell(180, 10, f"Analysis Time: {time.strftime('%H:%M:%S')}", ln=True, align="C")
                
                # Add basic match info
                pdf.ln(20)
                pdf.set_font("Arial", "B", 14)
                pdf.cell(180, 10, "Match Information", ln=True)
                
                pdf.set_font("Arial", "", 12)
                pdf.cell(90, 10, f"Home Team: {self.team_home}", ln=0)
                pdf.cell(90, 10, f"Away Team: {self.team_away}", ln=1)
                pdf.cell(90, 10, f"Home Style: {self.home_playstyle}", ln=0)
                pdf.cell(90, 10, f"Away Style: {self.away_playstyle}", ln=1)
                pdf.cell(180, 10, f"Video Duration: {self.video_info.duration:.2f} seconds", ln=1)
                pdf.cell(180, 10, f"Frames Analyzed: {self.video_info.processed_frames}", ln=1)
                
                # ---- Team Statistics Page ----
                pdf.add_page()
                add_section_header(pdf, "Team Statistics")
                
                # Team comparison table
                pdf.set_font("Arial", "B", 12)
                pdf.cell(60, 10, "Metric", border=1)
                pdf.cell(60, 10, self.team_home, border=1)
                pdf.cell(60, 10, self.team_away, border=1, ln=True)
                
                pdf.set_font("Arial", "", 10)
                
                # Add team stats
                metrics = [
                    "Possession (%)", 
                    "Distance (m)", 
                    "Passes",
                    "Pass Completion (%)",
                    "Forward Passes (%)",
                    "Progressive Passes",
                    "Breaking Lines Passes",
                    "Danger Zone Passes",
                    "Expected Assists (xA)",
                    "Shots", 
                    "Shots on Target",
                    "Shot Accuracy (%)",
                    "Expected Goals (xG)",
                    "Most Used Formation"
                ]
                
                for metric in metrics:
                    pdf.cell(60, 8, metric, border=1)
                    
                    if metric == "Most Used Formation":
                        pdf.cell(60, 8, str(self.team_stats[self.team_home]["Most Used Formation"]), border=1)
                        pdf.cell(60, 8, str(self.team_stats[self.team_away]["Most Used Formation"]), border=1, ln=True)
                    else:
                        pdf.cell(60, 8, str(self.team_stats[self.team_home].get(metric, "N/A")), border=1)
                        pdf.cell(60, 8, str(self.team_stats[self.team_away].get(metric, "N/A")), border=1, ln=True)
                
                # ---- Strengths and Weaknesses Page ----
                pdf.add_page()
                add_section_header(pdf, "Team Strengths & Weaknesses")
                
                # Home team
                pdf.set_font("Arial", "B", 14)
                pdf.cell(180, 10, f"{self.team_home} Analysis", ln=True)
                
                # Strengths
                pdf.set_font("Arial", "B", 12)
                pdf.cell(180, 8, "Strengths:", ln=True)
                pdf.set_font("Arial", "", 10)
                
                for key, data in self.team_strengths[self.team_home].items():
                    description = data["description"]
                    pdf.cell(180, 8, f"- {description}", ln=True)
                
                # Weaknesses
                pdf.set_font("Arial", "B", 12)
                pdf.ln(5)
                pdf.cell(180, 8, "Areas for Improvement:", ln=True)
                pdf.set_font("Arial", "", 10)
                
                for key, data in self.team_weaknesses[self.team_home].items():
                    description = data["description"]
                    pdf.cell(180, 8, f"- {description}", ln=True)
                
                # Away team
                pdf.ln(10)
                pdf.set_font("Arial", "B", 14)
                pdf.cell(180, 10, f"{self.team_away} Analysis", ln=True)
                
                # Strengths
                pdf.set_font("Arial", "B", 12)
                pdf.cell(180, 8, "Strengths:", ln=True)
                pdf.set_font("Arial", "", 10)
                
                for key, data in self.team_strengths[self.team_away].items():
                    description = data["description"]
                    pdf.cell(180, 8, f"- {description}", ln=True)
                
                # Weaknesses
                pdf.set_font("Arial", "B", 12)
                pdf.ln(5)
                pdf.cell(180, 8, "Areas for Improvement:", ln=True)
                pdf.set_font("Arial", "", 10)
                
                for key, data in self.team_weaknesses[self.team_away].items():
                    description = data["description"]
                    pdf.cell(180, 8, f"- {description}", ln=True)
                
                # ---- Tactical Suggestions Page ----
                pdf.add_page()
                add_section_header(pdf, "Tactical Suggestions")
                
                # Home team suggestions
                pdf.set_font("Arial", "B", 14)
                pdf.cell(180, 10, f"Tactical Suggestions for {self.team_home}", ln=True)
                pdf.set_font("Arial", "", 10)
                
                for i, suggestion in enumerate(self.tactical_suggestions[self.team_home]):
                    pdf.multi_cell(180, 8, f"{i+1}. {suggestion}", ln=True)
                    pdf.ln(2)
                
                # Away team suggestions
                pdf.ln(10)
                pdf.set_font("Arial", "B", 14)
                pdf.cell(180, 10, f"Tactical Suggestions for {self.team_away}", ln=True)
                pdf.set_font("Arial", "", 10)
                
                for i, suggestion in enumerate(self.tactical_suggestions[self.team_away]):
                    pdf.multi_cell(180, 8, f"{i+1}. {suggestion}", ln=True)
                    pdf.ln(2)
                
                # ---- Player Statistics Page ----
                if not self.player_stats_df.empty:
                    pdf.add_page()
                    add_section_header(pdf, "Player Statistics")
                    
                    # Home team players
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(180, 10, f"{self.team_home} Players", ln=True)
                    
                    home_players = self.player_stats_df[self.player_stats_df['Team'] == self.team_home].sort_values(by='Distance (m)', ascending=False)
                    
                    if not home_players.empty:
                        # Table header
                        pdf.set_font("Arial", "B", 8)
                        pdf.cell(15, 8, "ID", border=1)
                        pdf.cell(25, 8, "Distance (m)", border=1)
                        pdf.cell(20, 8, "Passes", border=1)
                        pdf.cell(25, 8, "Prog. Passes", border=1)
                        pdf.cell(20, 8, "Shots", border=1)
                        pdf.cell(35, 8, "xG", border=1)
                        pdf.cell(35, 8, "xA", border=1, ln=True)
                        
                        # Table content
                        pdf.set_font("Arial", "", 8)
                        for idx, row in home_players.iterrows():
                            pdf.cell(15, 8, str(row['Player ID']), border=1)
                            pdf.cell(25, 8, str(row['Distance (m)']), border=1)
                            pdf.cell(20, 8, str(row['Passes']), border=1)
                            prog_passes = row.get('Progressive Passes', 'N/A')
                            pdf.cell(25, 8, str(prog_passes), border=1)
                            pdf.cell(20, 8, str(row['Shots']), border=1)
                            xg = row.get('Expected Goals (xG)', 'N/A')
                            pdf.cell(35, 8, str(xg), border=1)
                            xa = row.get('Expected Assists (xA)', 'N/A')
                            pdf.cell(35, 8, str(xa), border=1, ln=True)
                    
                    # Away team players
                    pdf.ln(10)
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(180, 10, f"{self.team_away} Players", ln=True)
                    
                    away_players = self.player_stats_df[self.player_stats_df['Team'] == self.team_away].sort_values(by='Distance (m)', ascending=False)
                    
                    if not away_players.empty:
                        # Table header
                        pdf.set_font("Arial", "B", 8)
                        pdf.cell(15, 8, "ID", border=1)
                        pdf.cell(25, 8, "Distance (m)", border=1)
                        pdf.cell(20, 8, "Passes", border=1)
                        pdf.cell(25, 8, "Prog. Passes", border=1)
                        pdf.cell(20, 8, "Shots", border=1)
                        pdf.cell(35, 8, "xG", border=1)
                        pdf.cell(35, 8, "xA", border=1, ln=True)
                        
                        # Table content
                        pdf.set_font("Arial", "", 8)
                        for idx, row in away_players.iterrows():
                            pdf.cell(15, 8, str(row['Player ID']), border=1)
                            pdf.cell(25, 8, str(row['Distance (m)']), border=1)
                            pdf.cell(20, 8, str(row['Passes']), border=1)
                            prog_passes = row.get('Progressive Passes', 'N/A')
                            pdf.cell(25, 8, str(prog_passes), border=1)
                            pdf.cell(20, 8, str(row['Shots']), border=1)
                            xg = row.get('Expected Goals (xG)', 'N/A')
                            pdf.cell(35, 8, str(xg), border=1)
                            xa = row.get('Expected Assists (xA)', 'N/A')
                            pdf.cell(35, 8, str(xa), border=1, ln=True)
                
                # ---- Key Events Page ----
                if self.events:
                    pdf.add_page()
                    add_section_header(pdf, "Key Match Events")
                    
                    # Group events by 15-minute intervals
                    events_by_interval = {}
                    interval_size = 15 * 60  # 15 minutes in seconds
                    
                    for event in sorted(self.events, key=lambda e: e['time']):
                        interval = int(event['time'] // interval_size)
                        interval_str = f"{interval * 15}-{(interval + 1) * 15} min"
                        
                        if interval_str not in events_by_interval:
                            events_by_interval[interval_str] = []
                        
                        events_by_interval[interval_str].append(event)
                    
                    # Add events by interval
                    for interval_str, interval_events in events_by_interval.items():
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(180, 10, f"Events ({interval_str})", ln=True)
                        
                        pdf.set_font("Arial", "", 10)
                        for event in interval_events:
                            time_str = f"{int(event['time'] // 60)}:{int(event['time'] % 60):02d}"
                            team = event['team']
                            
                            if event['type'] == 'pass':
                                # Enhanced pass description
                                direction = event.get('direction', 'unknown')
                                pass_type = event.get('pass_type', 'normal')
                                progressive = "progressive " if event.get('progressive', False) else ""
                                danger_zone = "into danger zone " if event.get('danger_zone', False) else ""
                                
                                pdf.multi_cell(180, 8, f"{time_str} - {progressive}{danger_zone}{pass_type} {direction} pass from Player {event['from_player']} to Player {event['to_player']} ({team})")
                            
                            elif event['type'] == 'shot':
                                # Enhanced shot description
                                on_target = "on target" if event.get('on_target', False) else "off target"
                                xg = f" (xG: {event.get('xG', 0):.2f})" if 'xG' in event else ""
                                
                                pdf.multi_cell(180, 8, f"{time_str} - Shot {on_target} by Player {event['player']} ({team}) at {event['target_goal']} goal{xg}")
                        
                        pdf.ln(5)
                
                # ---- AI Insights Page (if available) ----
                if self.enable_gemini and (self.gemini_insights[self.team_home] or self.gemini_insights[self.team_away]):
                    pdf.add_page()
                    add_section_header(pdf, "AI-Generated Insights")
                    
                    # Add AI insights with formatting
                    pdf.set_font("Arial", "", 10)
                    
                    insights_added = 0
                    for insight in self.gemini_insights[self.team_home]:
                        if "key tactical observations" in insight.lower():
                            # Get each line and format it properly
                            lines = insight.split('\n')
                            
                            # Add the heading
                            pdf.set_font("Arial", "B", 12)
                            if lines and lines[0].strip():
                                pdf.cell(180, 10, lines[0].replace('#', '').strip(), ln=True)
                            
                            # Add the content
                            pdf.set_font("Arial", "", 10)
                            for line in lines[1:]:
                                line = line.strip()
                                if line:
                                    pdf.multi_cell(180, 6, line)
                            
                            pdf.ln(5)
                            insights_added += 1
                            
                            # Limit to a reasonable number of insights
                            if insights_added >= 2:
                                break
                
                # ---- Final page with credits ----
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(180, 10, "Analysis Summary", ln=True)
                
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(180, 8, f"This report presents a comprehensive analysis of the match between {self.team_home} and {self.team_away}. The analysis includes team statistics, player performance metrics, key events, tactical observations, and personalized suggestions for both teams.")
                pdf.ln(5)
                
                pdf.multi_cell(180, 8, "The analysis is based on computer vision and AI-powered tracking of players and ball movement throughout the match. Statistical analysis was performed to identify patterns, strengths, and areas for improvement.")
                pdf.ln(10)
                
                pdf.set_font("Arial", "I", 10)
                pdf.cell(180, 8, "Report generated using Advanced Football Analysis Platform", ln=True, align="C")
                pdf.cell(180, 8, f"Generated on {time.strftime('%Y-%m-%d at %H:%M:%S')}", ln=True, align="C")
                
                # Save PDF
                report_path = "enhanced_match_analysis_report.pdf"
                pdf.output(report_path)
                
                # Provide download link
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Enhanced Analysis Report",
                        data=file,
                        file_name="enhanced_match_analysis_report.pdf",
                        mime="application/pdf"
                    )
                    
                st.success("✅ Comprehensive match analysis report generated successfully!")
                
                # Report preview
                st.subheader("Report Preview")
                
                # PDF preview thumbnail
                st.markdown("""
                <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center;">
                    <h3>Enhanced Match Analysis Report</h3>
                    <p style="color: #666;">{} vs {}</p>
                    <p>This PDF report includes:</p>
                    <ul style="text-align: left; display: inline-block;">
                        <li>Team statistics comparison</li>
                        <li>Team strengths and weaknesses analysis</li>
                        <li>Tactical suggestions for both teams</li>
                        <li>Player performance data</li>
                        <li>Key match events timeline</li>
                        <li>AI-generated insights</li>
                    </ul>
                </div>
                """.format(self.team_home, self.team_away), unsafe_allow_html=True)
                
                # Report contents
                with st.expander("Report Table of Contents"):
                    st.markdown("""
                    1. **Cover Page**
                       - Match details
                       - Analysis information
                       
                    2. **Team Statistics**
                       - Possession and movement stats
                       - Passing metrics
                       - Shot analysis
                       
                    3. **Team Strengths & Weaknesses**
                       - Key strengths of both teams
                       - Areas for improvement
                       
                    4. **Tactical Suggestions**
                       - Tactical recommendations for both teams
                       
                    5. **Player Statistics**
                       - Individual player metrics
                       - Performance comparison
                       
                    6. **Key Match Events**
                       - Timeline of important match events
                       
                    7. **AI-Generated Insights**
                       - Advanced tactical observations
                       
                    8. **Summary**
                       - Overall match analysis
                    """)
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            traceback.print_exc()
    
    def run(self):
        """Main method to run the football analysis"""
        try:
            # Check if video info is available in session state
            if st.session_state[KEY_VIDEO_INFO]:
                self.video_info = st.session_state[KEY_VIDEO_INFO]
            
            # If analysis is already complete, just display the results
            if st.session_state[KEY_ANALYSIS_COMPLETE]:
                if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                    self.display_video_analysis(st.session_state.processed_video_path)
                    self.display_player_stats()
                    self.display_spatial_analysis()
                    self.display_team_analysis()
                    
                    # Display strengths, weaknesses and tactical suggestions
                    if self.enable_tactical:
                        self.display_strengths_weaknesses()
                        self.display_tactical_suggestions()
                    
                    # Generate report if enabled
                    if self.enable_report:
                        self.generate_report()
                else:
                    st.error("Analysis results are incomplete or video file is missing. Please reset and try again.")
                    if st.button("Reset Analysis"):
                        for key in [KEY_ANALYSIS_COMPLETE, KEY_PLAYER_STATS, KEY_TEAM_STATS, KEY_EVENTS]:
                            st.session_state[key] = None
                        st.session_state[KEY_VIDEO_INFO] = VideoInfo()
                        st.session_state[KEY_ANALYSIS_COMPLETE] = False
                        st.session_state.processed_video_path = None
                        st.experimental_rerun()
                return
                
            # If start analysis button is clicked and video is uploaded
            if self.start_analysis and self.video_path is not None:
                with st.spinner("Processing video and analyzing football match..."):
                    output_video_path = self.process_video()
                    
                if output_video_path and os.path.exists(output_video_path):
                    # Analyze strengths and weaknesses
                    with st.spinner("Analyzing team strengths and weaknesses..."):
                        self.analyze_strengths_weaknesses()
                    
                    # Display results in tabs
                    self.display_video_analysis(output_video_path)
                    self.display_player_stats()
                    self.display_spatial_analysis()
                    self.display_team_analysis()
                    
                    # Display strengths, weaknesses and tactical suggestions
                    if self.enable_tactical:
                        self.display_strengths_weaknesses()
                        self.display_tactical_suggestions()
                    
                    # Generate report if enabled
                    if self.enable_report:
                        self.generate_report()
                    
                    st.sidebar.success("✅ Analysis completed successfully!")
                    
                    # Allow download of output video
                    with open(output_video_path, "rb") as file:
                        st.sidebar.download_button(
                            label="📥 Download Processed Video",
                            data=file,
                            file_name="football_analysis_video.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("Video processing failed. Please check the logs and try again.")
            else:
                st.info("Upload a football match video and click 'Start Analysis' to begin.")
                
                # Show sample images of what the analysis will provide
                st.subheader("👁️ Preview of Analysis Capabilities")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### Enhanced Player Tracking")
                    st.markdown("""
                    The system tracks players across frames with:
                    - Team identification
                    - Position tracking
                    - Movement analysis
                    - Speed and acceleration metrics
                    - Pass network visualization
                    """)
                    
                with col2:
                    st.markdown("#### Advanced Analytics")
                    st.markdown("""
                    Advanced analysis features include:
                    - Heatmaps for player movement
                    - Enhanced pass and shot analysis
                    - Possession zones
                    - Advanced event detection
                    - Pass and shot quality metrics (xG, xA)
                    - Comprehensive team statistics
                    """)
                    
                with col3:
                    st.markdown("#### AI Tactical Analysis")
                    st.markdown("""
                    AI-powered tactical features:
                    - Team strengths & weaknesses detection
                    - Detailed tactical suggestions
                    - Opponent playstyle analysis
                    - Strategic recommendations
                    - SWOT analysis for each team
                    - Gemini AI integration for expert insights
                    """)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            traceback.print_exc()
            
            # Offer reset option
            if st.button("Reset Application"):
                for key in [KEY_ANALYSIS_COMPLETE, KEY_PLAYER_STATS, KEY_TEAM_STATS, KEY_EVENTS]:
                    st.session_state[key] = None
                st.session_state[KEY_VIDEO_INFO] = VideoInfo()
                st.session_state[KEY_ANALYSIS_COMPLETE] = False
                st.session_state.processed_video_path = None
                st.experimental_rerun()

# Run the application
if __name__ == "__main__":
    try:
        # Set page configuration
        analyzer = FootballAnalyzer()
        analyzer.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        traceback.print_exc()