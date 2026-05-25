import base64
import os
from mcp_yolo_server import detect

def run_local_test(image_path: str):
    """
    Test the YOLO detection tool and save the resulting annotated image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    print(f"Testing detection on: {image_path}")
    print("Note: Detection is performed only on the rightmost 1/3 of the image.")
    
    # Call the detect tool
    # This will load the YOLO model (via the import) and process the image
    results = detect(image_path)

    # results[0] is TextContent (JSON and summary)
    # results[1] is ImageContent (Base64 encoded annotated image)
    text_content = results[0].text
    image_data_b64 = results[1].data

    print("\n--- Detection Text Output ---")
    print(text_content)

    # Decode and save the annotated image
    output_path = "detection_result.jpg"
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(image_data_b64))
    
    print(f"\nAnnotated image saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    # Update this path to an actual image file you want to test
    TEST_IMAGE = r"C:\Users\lsant\Downloads\Final_sem_project\Final_dataset\bogus\fuse_bogus_full\images\011995.png" 
    run_local_test(TEST_IMAGE)
