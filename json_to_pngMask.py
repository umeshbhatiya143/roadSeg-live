import labelme
import os
import numpy as np
from PIL import Image, ImageDraw

def create_masks_from_json_folder(json_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(json_folder):
        if file_name.endswith(".json"):
            json_path = os.path.join(json_folder, file_name)
            print(f"Processing {json_path}...")

            try:
                data = labelme.LabelFile(json_path)
                shapes = data.shapes

                # Get image as numpy array
                img_data = data.imageData
                if img_data is None:
                    # if imageData is None, load from imagePath
                    img_path = os.path.join(os.path.dirname(json_path), data.imagePath)
                    img_array = np.array(Image.open(img_path))
                else:
                    img_array = labelme.utils.img_data_to_arr(img_data)

                img_height, img_width = img_array.shape[:2]

                # Empty masks
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                mask_viz = np.zeros((img_height, img_width, 3), dtype=np.uint8)

                # Label and color mapping
                label_mapping = {}
                color_mapping = {}
                current_label = 1

                for shape in shapes:
                    label = shape['label']
                    if label not in label_mapping:
                        label_mapping[label] = current_label
                        color_mapping[label] = tuple(np.random.randint(0, 256, size=3).tolist())
                        current_label += 1

                    points = np.array(shape['points'], dtype=np.int32)
                    pil_mask = Image.fromarray(mask)
                    ImageDraw.Draw(pil_mask).polygon([tuple(p) for p in points],
                                                     outline=label_mapping[label],
                                                     fill=label_mapping[label])
                    mask = np.array(pil_mask)

                    pil_viz = Image.fromarray(mask_viz)
                    ImageDraw.Draw(pil_viz).polygon([tuple(p) for p in points],
                                                    outline=color_mapping[label],
                                                    fill=color_mapping[label])
                    mask_viz = np.array(pil_viz)

                # Save masks
                base_name = os.path.splitext(file_name)[0]
                mask_path = os.path.join(output_folder, base_name + ".png")
                mask_viz_path = os.path.join(output_folder, base_name + "_mask_viz.png")
                Image.fromarray(mask).save(mask_path)
                Image.fromarray(mask_viz).save(mask_viz_path)

                print(f"Saved: {mask_path} and {mask_viz_path}")

            except Exception as e:
                print(f"Error processing {json_path}: {e}")

if __name__ == "__main__":
    json_folder = "/home/umesh/Desktop/Road_segmentation/dataset/d"
    output_folder = "/home/umesh/Desktop/Road_segmentation/dataset/mask_masks"
    create_masks_from_json_folder(json_folder, output_folder)
