import gradio as gr
from diffusers import StableDiffusionUpscalePipeline
import torch
from PIL import Image
import os

# Custom CSS for Ritika and Lata theme
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Lora:ital@0;1&display=swap');

:root {
    --primary-color: #8a7090;
    --secondary-color: #6b4e71;
    --background-start: #f5e6f2;
    --background-end: #e0d8f5;
    --text-color: #4a4a4a;
    --shadow-color: rgba(0, 0, 0, 0.1);
}

body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, var(--background-start), var(--background-end));
    color: var(--text-color);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    box-shadow: 0 4px 15px var(--shadow-color);
}

h1 {
    font-family: 'Lora', serif;
    color: var(--secondary-color);
    font-size: 3em;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px var(--shadow-color);
}

.tagline {
    font-family: 'Lora', serif;
    color: var(--primary-color);
    font-style: italic;
    font-size: 1.2em;
    margin-top: 0.5rem;
}

.gr-button {
    background-color: var(--primary-color) !important;
    border: none !important;
    border-radius: 25px !important;
    color: white !important;
    padding: 12px 24px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px var(--shadow-color) !important;
}

.gr-button:hover {
    background-color: var(--secondary-color) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 8px var(--shadow-color) !important;
}

.gr-box {
    border-radius: 15px !important;
    background: rgba(255, 255, 255, 0.9) !important;
    box-shadow: 0 4px 15px var(--shadow-color) !important;
    padding: 20px !important;
}

.loading {
    display: none;
    text-align: center;
    margin: 20px 0;
}

.loading.active {
    display: block;
}

.loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid var(--background-start);
    border-top: 4px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
"""

class ImageUpscaler:
    def __init__(self):
        print("Loading model... This might take a few minutes.")
        self.model_id = "stabilityai/stable-diffusion-x4-upscaler"
        
        # Check if CUDA is available and set device accordingly
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model with better memory handling for laptop GPUs
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for faster processing
            low_memory=True  # Enable low memory optimizations
        )
        
        # Move model to appropriate device
        if self.device == "cuda":
            # Enable all memory optimizations for CUDA
            self.pipeline.enable_attention_slicing(1)  # Maximum memory savings
            self.pipeline.enable_sequential_cpu_offload()  # Better than model_cpu_offload for laptops
            self.pipeline.enable_vae_slicing()  # Enable VAE slicing for memory efficiency
        
        self.pipeline = self.pipeline.to(self.device)
        print("Model loaded successfully!")

    def upscale_image(self, image):
        if image is None:
            return None, "Please upload an image."
        
        try:
            # Convert image to RGB and resize if too large
            low_res_img = image.convert("RGB")
            
            # Get image dimensions
            width, height = low_res_img.size
            
            # More conservative max size for laptop GPUs
            max_size = 384  # Reduced from 512 for better memory handling
            if width > max_size or height > max_size:
                ratio = min(max_size/width, max_size/height)
                new_size = (int(width*ratio), int(height*ratio))
                low_res_img = low_res_img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Resized image from {width}x{height} to {new_size[0]}x{new_size[1]}")
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()  # Reset memory stats
            
            # Perform upscaling with optimized parameters for laptops
            upscaled_image = self.pipeline(
                prompt="enhance quality, sharp details, high resolution photograph",
                image=low_res_img,
                num_inference_steps=15,  # Further reduced steps for laptop GPUs
                guidance_scale=7.0,  # Slightly reduced guidance scale
                generator=torch.manual_seed(42)  # Fixed seed for consistent results
            ).images[0]
            
            # Force garbage collection after processing
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return upscaled_image, "Image upscaled successfully!"
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return None, "Out of memory error. Please try with a smaller image or restart the application."
            return None, f"Runtime error during upscaling: {str(e)}"
        except Exception as e:
            return None, f"Error during upscaling: {str(e)}"

def create_interface():
    upscaler = ImageUpscaler()
    
    with gr.Blocks(css=CUSTOM_CSS) as interface:
        # Header
        gr.HTML("""
            <div class="header">
                <h1>Sharpify</h1>
                <div class="tagline">Enhance Your Moments with Ritika and Lata</div>
            </div>
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil"
                )
                upscale_button = gr.Button("✨ Enhance Image", variant="primary")
                
            with gr.Column():
                output_image = gr.Image(label="Enhanced Image")
                status_text = gr.Textbox(label="Status", interactive=False, value="Ready")
                progress = gr.HTML("""
                    <div class="progress-container" style="display: none;">
                        <div class="progress-bar"></div>
                        <div class="progress-text">Processing... Please wait.</div>
                    </div>
                """)
        
        # Loading animation with better visibility
        gr.HTML("""
            <style>
                .progress-container {
                    width: 100%;
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 8px;
                    background: rgba(255, 255, 255, 0.1);
                }
                .progress-bar {
                    width: 0%;
                    height: 4px;
                    background: var(--primary-color);
                    border-radius: 2px;
                    transition: width 0.3s ease;
                    animation: progress 1s infinite ease-in-out;
                }
                .progress-text {
                    margin-top: 8px;
                    text-align: center;
                    color: var(--primary-color);
                    font-size: 14px;
                }
                @keyframes progress {
                    0% { width: 20%; }
                    50% { width: 80%; }
                    100% { width: 20%; }
                }
                .processing {
                    animation: pulse 1.5s infinite;
                }
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.6; }
                    100% { opacity: 1; }
                }
            </style>
        """)
        
        def process_image(image):
            status_text.value = "Processing..."
            return upscaler.upscale_image(image)
        
        # Handle upscaling with better loading state
        upscale_button.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[output_image, status_text],
            js="""
                async (input_image) => {
                    if (!input_image) {
                        return;
                    }
                    
                    // Show progress and update status
                    const progress = document.querySelector('.progress-container');
                    const status = document.querySelector('.status');
                    const button = document.querySelector('button.primary');
                    
                    if (progress) progress.style.display = 'block';
                    if (status) status.textContent = 'Processing...';
                    if (button) {
                        button.disabled = true;
                        button.classList.add('processing');
                    }
                    
                    try {
                        return input_image;
                    } finally {
                        setTimeout(() => {
                            if (progress) progress.style.display = 'none';
                            if (button) {
                                button.disabled = false;
                                button.classList.remove('processing');
                            }
                        }, 500);
                    }
                }
            """
        )
        
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.queue()  # Enable queuing for better handling of multiple requests
    demo.launch(share=True, debug=True)  # Enable debug mode for better error reporting 