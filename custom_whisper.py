import os
import sys
import whisper
from datetime import datetime
import subprocess
import argparse
from pathlib import Path

def create_datetime_directory(base_dir="output"):
    """Create directory structure based on current date/time: YY/MM/ddhh/"""
    now = datetime.now()
    year_month = f"{now.year % 100:02d}/{now.month:02d}"
    day_hour = f"{now.day:02d}{now.hour:02d}"
    
    # Create the directory path
    dir_path = os.path.join(base_dir, year_month, day_hour)
    os.makedirs(dir_path, exist_ok=True)
    
    # Return the directory path and minutes/seconds for filename
    return dir_path, f"{now.minute:02d}{now.second:02d}"

def transcribe_audio(audio_path, model_name="tiny", output_base_dir="output", language=None, task="transcribe"):
    """Transcribe audio file and save to datetime-based directory structure"""
    try:
        # Create directory structure based on current date/time
        output_dir, time_suffix = create_datetime_directory(output_base_dir)
        
        # Load the Whisper model
        print(f"Loading model: {model_name}")
        model = whisper.load_model(model_name)
        
        # Transcribe the audio
        print(f"Transcribing: {audio_path}")
        transcribe_options = {}
        if language:
            transcribe_options["language"] = language
        if task:
            transcribe_options["task"] = task
        
        result = model.transcribe(audio_path, **transcribe_options)
        
        # Save the transcription to a text file with minute/second as filename
        output_file = os.path.join(output_dir, f"{time_suffix}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        print(f"Transcription saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper with date/time-based output structure")
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--model", default="tiny", type=str, help="Whisper model to use (tiny, base, small, medium, large, turbo)")
    parser.add_argument("--output", default="output", type=str, help="Base output directory")
    parser.add_argument("--language", type=str, default=None, help="Language of the audio (if not provided, Whisper will detect)")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], 
                        help="Task to perform (transcribe or translate to English)")
    
    args = parser.parse_args()
    
    # Ensure the audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file {args.audio} does not exist")
        sys.exit(1)
    
    # Transcribe the audio
    output_file = transcribe_audio(
        args.audio, 
        model_name=args.model, 
        output_base_dir=args.output,
        language=args.language,
        task=args.task
    )
    
    if not output_file:
        sys.exit(1)

if __name__ == "__main__":
    main() 