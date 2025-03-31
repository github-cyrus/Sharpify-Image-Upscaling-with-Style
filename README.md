# Sharpify - Image Upscaling with Style

Sharpify is an elegant image upscaling application that uses the Stable Diffusion x4 Upscaler model to enhance your images. Built with a beautiful custom theme called "Ritika and Lata," it provides a user-friendly interface for image upscaling.

## Features

- 4x image upscaling using state-of-the-art Stable Diffusion technology
- Beautiful custom "Ritika and Lata" theme with elegant design
- User-friendly web interface built with Gradio
- Real-time processing with loading animations
- Status feedback and error handling
- GPU acceleration support

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended) for faster processing
- Internet connection for model download

## Installation

1. Clone this repository or download the files
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to the URL shown in the terminal (usually http://127.0.0.1:7860)
3. Upload an image using the upload button
4. Click "Enhance Image" and wait for the processing to complete
5. The enhanced image will appear in the output section

## Notes

- The first time you run the application, it will download the model which may take a few minutes
- Processing time depends on your hardware and image size
- For optimal results, ensure input images are clear and well-lit

## Credits

- Stable Diffusion x4 Upscaler model by Stability AI
- Built with Gradio, PyTorch, and Diffusers
- Custom "Ritika and Lata" theme design

## License

MIT License - Feel free to use and modify as needed! 