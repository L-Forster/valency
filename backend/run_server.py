"""
Run the ToneAnalytics API server.

This script starts the FastAPI server for the ToneAnalytics API,
which provides endpoints for analyzing the emotional tone of text.
"""

import os
import sys
import uvicorn

def main():
    """Run the ToneAnalytics API server"""
    print("=" * 80)
    print("TONEANALYTICS API SERVER")
    print("=" * 80)
    
    # Check if models directory exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    models_dir = os.path.join(parent_dir, "models/paragraph")
    
    if not os.path.exists(models_dir):
        print(f"\nWARNING: Models directory '{models_dir}' does not exist.")
        print("The neural network models might not be available.")
        
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Exiting. Please ensure the models are available.")
            sys.exit(0)
    
    print("\nStarting ToneAnalytics API server...")
    print("\nAPI will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server")
    print("-" * 80)
    
    # Run the server
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=[parent_dir]
    )

if __name__ == "__main__":
    main() 