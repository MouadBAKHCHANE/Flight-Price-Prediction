import os
from PIL import Image

def convert_to_webp_recursive(root_dir):
    print(f"Scanning {root_dir} recursively...")
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter for images
        image_files = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            print(f"Found {len(image_files)} images in {dirpath}")
            
            for filename in image_files:
                try:
                    file_path = os.path.join(dirpath, filename)
                    file_name_no_ext = os.path.splitext(filename)[0]
                    webp_path = os.path.join(dirpath, f"{file_name_no_ext}.webp")
                    
                    # Skip if already exists
                    if os.path.exists(webp_path):
                        print(f"Skipping {filename} (WebP exists)")
                        continue

                    # Convert
                    with Image.open(file_path) as img:
                        img.save(webp_path, "WEBP")
                    
                    print(f"Converted: {filename}")
                except Exception as e:
                    print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    target_dir = r"c:/Users/EliteBook/Downloads/DATA/Projects/Flight Price Prediction"
    convert_to_webp_recursive(target_dir)
