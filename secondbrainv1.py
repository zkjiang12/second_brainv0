import google.genai as genai
from google.genai import types
import os
from dotenv import load_dotenv
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def split_video_into_clips(input_video, output_dir="clips", segment_duration=30):
    """
    Split a video into multiple clips of specified duration using ffmpeg
    
    Args:
        input_video (str): Path to the input video file
        output_dir (str): Directory to save the clips
        segment_duration (int): Duration of each clip in seconds
    
    Returns:
        list: List of created clip filenames
    """
    
    # Check if input video exists
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video '{input_video}' not found")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get the base name of the input video (without extension)
    base_name = Path(input_video).stem
    
    # Define output pattern
    output_pattern = os.path.join(output_dir, f"{base_name}_part_%03d.mp4")
    
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-c", "copy",  # Copy streams without re-encoding (faster)
        "-map", "0",   # Map all streams from input
        "-segment_time", str(segment_duration),
        "-f", "segment",
        "-reset_timestamps", "1",
        "-y",  # Overwrite output files without asking
        output_pattern
    ]
    
    print(f"Splitting '{input_video}' into {segment_duration}-second clips...")
    print(f"Output directory: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run ffmpeg command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Get list of created files
        clip_files = []
        for file in os.listdir(output_dir):
            if file.startswith(f"{base_name}_part_") and file.endswith(".mp4"):
                clip_files.append(os.path.join(output_dir, file))
        
        clip_files.sort()  # Sort to ensure correct order
        
        print(f"Successfully created {len(clip_files)} clips:")
        for clip in clip_files:
            print(f"  - {clip}")
        
        return clip_files
        
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        print(f"stderr: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def process_single_clip_with_gemini(clip_path, client):
    """
    Process a single video clip with Gemini
    
    Args:
        clip_path (str): Path to the video clip
        client: Gemini client instance
    
    Returns:
        dict: Contains clip_path, response_text, and processing_time
    """
    
    start_time = time.time()
    
    try:
        # Read video bytes
        with open(clip_path, 'rb') as f:
            video_bytes = f.read()
        
        # Generate content with Gemini
        response = client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                    ),
                    types.Part(text=
                        """
                        
                        CONTEXT
                            You are analyzing video footage from smart glasses worn by someone building an AI second brain. Your job is to understand what they're working on or learning and proactively enhance their knowledge.
                            YOUR ROLE
                            Identify the user's work/learning activities and provide intelligent research and insights to deepen their understanding and accelerate their progress.
                        
                        CORE TASKS
                        1. ACTIVITY SUMMARY

                            What is the user working on, reading, or studying?
                            What key topics, concepts, or problems are they engaging with?
                            What seems to be their current focus or challenge?

                        2. PROACTIVE RESEARCH & INSIGHTS

                            For identified activities/topics, provide:

                            Related concepts: Connected ideas they should know about
                            Deeper research: Latest developments, key papers, or advanced topics
                            Practical applications: How this knowledge can be applied
                            Interesting connections: Cross-disciplinary insights or unexpected angles
                            Expert perspectives: What leading practitioners/researchers are saying

                        OUTPUT FORMAT
                        SUMMARY
                            Brief overview of what the user was working on or learning.
                        
                        RESEARCH & INSIGHTS
                            For each topic/activity identified provide additional helpful insights/important information for the user:
                            Topic: What they were engaged with
                            Key Insights: Relevant research, developments, or expert knowledge
                            Connections: How this relates to other fields or concepts they care about
                            Next Level: Advanced concepts or applications to explore
                            Interesting Angles: Unexpected perspectives or emerging trends
                        """
                    )
                ]
            )
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'clip_path': clip_path,
            'response_text': response.text,
            'processing_time': processing_time,
            'success': True
        }
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'clip_path': clip_path,
            'error': str(e),
            'processing_time': processing_time,
            'success': False
        }

def process_clips_in_parallel(clips, client, max_workers=5):
    """
    Process multiple video clips in parallel using Gemini
    
    Args:
        clips (list): List of clip file paths
        client: Gemini client instance
        max_workers (int): Maximum number of parallel workers
    
    Returns:
        list: List of results from each clip processing
    """
    
    print(f"Processing {len(clips)} clips in parallel with {max_workers} workers...")
    
    results = []
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_clip = {
            executor.submit(process_single_clip_with_gemini, clip, client): clip 
            for clip in clips
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_clip):
            clip = future_to_clip[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    print(f"✅ Completed: {os.path.basename(clip)} ({result['processing_time']:.2f}s)")
                else:
                    print(f"❌ Failed: {os.path.basename(clip)} - {result['error']}")
                    
            except Exception as e:
                print(f"❌ Exception processing {clip}: {e}")
                results.append({
                    'clip_path': clip,
                    'error': str(e),
                    'success': False
                })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 Parallel processing completed in {total_time:.2f}s")
    print(f"Average per clip: {total_time/len(clips):.2f}s")
    
    return results

load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Split the video into clips
print("🎬 Splitting video into clips...")
clips = split_video_into_clips("glassesvid.mp4")

# Process all clips in parallel with Gemini
print("🧠 Processing clips with Gemini in parallel...")
results = process_clips_in_parallel(clips, client, max_workers=5)

# Print results
print("\n📊 RESULTS SUMMARY:")
successful_results = [r for r in results if r['success']]
failed_results = [r for r in results if not r['success']]

print(f"✅ Successfully processed: {len(successful_results)} clips")
print(f"❌ Failed to process: {len(failed_results)} clips")

if successful_results:
    total_processing_time = sum(r['processing_time'] for r in successful_results)
    print(f"⏱️ Total processing time: {total_processing_time:.2f}s")
    print(f"🚀 Average per clip: {total_processing_time/len(successful_results):.2f}s")

# Print individual results
print("\n📝 INDIVIDUAL CLIP ANALYSES:")
for i, result in enumerate(successful_results):
    print(f"\n{'='*60}")
    print(f"CLIP {i+1}: {os.path.basename(result['clip_path'])}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"{'='*60}")
    print(result['response_text'])
    print(f"{'='*60}")

# Print any errors
if failed_results:
    print("\n❌ FAILED CLIPS:")
    for result in failed_results:
        print(f"- {os.path.basename(result['clip_path'])}: {result['error']}")

