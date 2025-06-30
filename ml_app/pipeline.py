# === Main Execution ===
import supervision as sv
from classification import process_soccer_video
from config import SOURCE_VIDEO_PATH, OUTPUT_VIDEO_PATH

try:
    # Load models
    print("🔄 Loading detection models...")
       
    # Initialize annotators
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=20, height=17
    )
        
    # Process video with enhanced tracking
    print("\n🚀 Starting video processing...")
    process_soccer_video(
        source_path=SOURCE_VIDEO_PATH,
        output_path=OUTPUT_VIDEO_PATH,
    )
        
    print("\n🎉 Processing completed successfully!")
        
except Exception as e:
    print(f"❌ Fatal error: {e}")
    import traceback
    traceback.print_exc()