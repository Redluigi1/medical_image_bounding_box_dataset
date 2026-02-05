import os
from datetime import datetime
from pdf2image import convert_from_path

def convert_pdfs_to_jpegs(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_counter = 1


    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            
            print(f"Processing: {filename}...")
            
            try:
       
                images = convert_from_path(pdf_path)

                for i, image in enumerate(images):
                
                    output_filename = f"{timestamp}_{image_counter}.jpg"
                    save_path = os.path.join(output_folder, output_filename)
                    image.save(save_path, "JPEG")
                    image_counter += 1
                    
            except Exception as e:
                print(f"Could not convert {filename}: {e}")

    print(f"\nDone! All images saved to '{output_folder}'.")

if __name__ == "__main__":

    convert_pdfs_to_jpegs("docs", "raw images")