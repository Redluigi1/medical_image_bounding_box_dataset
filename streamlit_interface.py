import streamlit as st
import os
import json
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw


IMAGE_FOLDER = "raw_images"
ANNOTATION_FOLDER = "annotations"
os.makedirs(ANNOTATION_FOLDER, exist_ok=True)

st.set_page_config(layout="wide", page_title="Box Annotator")


if "img_idx" not in st.session_state:
    st.session_state.img_idx = 0
if "committed_boxes" not in st.session_state:
    st.session_state.committed_boxes = [] 
if "current_points" not in st.session_state:
    st.session_state.current_points = []

images = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if not images:
    st.error(f"No images found in '{IMAGE_FOLDER}' folder.")
    st.stop()

current_img_name = images[st.session_state.img_idx]
img_path = os.path.join(IMAGE_FOLDER, current_img_name)


st.sidebar.title("Controls")
st.sidebar.info(f"Image {st.session_state.img_idx + 1} / {len(images)}")
st.sidebar.write(f"**File:** {current_img_name}")

if st.sidebar.button("↩️ Undo Last Point"):
    if st.session_state.current_points:
        st.session_state.current_points.pop()
        st.rerun()

can_add = len(st.session_state.current_points) == 4
if st.sidebar.button("➕ Add This Box", disabled=not can_add, use_container_width=True):
 
    int_points = [(int(pt[0]), int(pt[1])) for pt in st.session_state.current_points]
    st.session_state.committed_boxes.append(int_points)
    st.session_state.current_points = []
    st.rerun()

st.sidebar.markdown("---")

st.sidebar.subheader("Jump to Image")
jump_col1, jump_col2 = st.sidebar.columns([2, 1])
jump_to = jump_col1.number_input(
    "Image #", 
    min_value=1, 
    max_value=len(images), 
    value=st.session_state.img_idx + 1,
    step=1,
    label_visibility="collapsed"
)
if jump_col2.button("🚀 Go"):
    target_idx = int(jump_to) - 1  # Convert to 0-indexed
    if 0 <= target_idx < len(images) and target_idx != st.session_state.img_idx:
        st.session_state.img_idx = target_idx
        st.session_state.committed_boxes = []
        st.session_state.current_points = []
        st.rerun()

st.sidebar.markdown("---")


col_skip, col_save = st.sidebar.columns(2)

if col_skip.button("⏭️ Skip Image"):
    if st.session_state.img_idx < len(images) - 1:
        st.session_state.img_idx += 1
        st.session_state.committed_boxes = []
        st.session_state.current_points = []
        st.rerun()
    else:
        st.success("End of folder reached.")

if col_save.button("💾 Save & Next"):
    json_path = os.path.join(ANNOTATION_FOLDER, f"{current_img_name}.json")
    output_data = {"filename": current_img_name, "boxes": st.session_state.committed_boxes}
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=4)
    
    if st.session_state.img_idx < len(images) - 1:
        st.session_state.img_idx += 1
        st.session_state.committed_boxes = []
        st.session_state.current_points = []
        st.rerun()
    else:
        st.success("All images finished!")

img = Image.open(img_path)
draw = ImageDraw.Draw(img)


for box in st.session_state.committed_boxes:

    draw.polygon(box, outline="lime", width=10)


for i, pt in enumerate(st.session_state.current_points):
    x, y = int(pt[0]), int(pt[1])
    draw.ellipse([x-12, y-12, x+12, y+12], fill="red")
    draw.text((x+15, y), str(i+1), fill="yellow")


value = streamlit_image_coordinates(img, key="coords")

if value:

    new_pt = (int(value["x"]), int(value["y"]))
    if not st.session_state.current_points or new_pt != st.session_state.current_points[-1]:
        if len(st.session_state.current_points) < 4:
            st.session_state.current_points.append(new_pt)
            st.rerun()