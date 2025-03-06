import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from matplotlib.widgets import Button, RadioButtons

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print("\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
          "give numerically different outputs and sometimes degraded performance on MPS.")
np.random.seed(3)


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def load_sam2_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sam2_checkpoint = os.path.join(current_dir, "models", "sam2.1_hiera_large.pt")
    model_cfg = os.path.join(current_dir, "models", "sam2.1_hiera_l.yaml")

    if not os.path.exists(sam2_checkpoint):
        print(f"Error: Checkpoint file not found at {sam2_checkpoint}")
        print(f"Available files: {os.listdir(os.path.join(current_dir, 'models'))}")
        raise FileNotFoundError(f"Checkpoint file not found: {sam2_checkpoint}")

    if not os.path.exists(model_cfg):
        yaml_files = [f for f in os.listdir(os.path.join(current_dir, 'models')) if f.endswith('.yaml')]
        if not yaml_files:
            raise FileNotFoundError(f"No YAML configuration files found in {os.path.join(current_dir, 'models')}")
        model_cfg = os.path.join(current_dir, "models", yaml_files[0])
        print(f"Using alternative configuration file: {model_cfg}")

    print(f"Loading model with:\n- Checkpoint: {sam2_checkpoint}\n- Config: {model_cfg}")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


def capture_clicks_on_image():
    image_path = input("Enter the full path to the image: ")
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return

    try:
        image_pil = Image.open(image_path)
        image_RGB = np.array(image_pil.convert("RGB"))
        image = plt.imread(image_path)
        print(f"Image loaded successfully: shape {image.shape}, type {image.dtype}")
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return

    # Variables for storing coordinates
    click_coords = []
    box_coords = []
    drawing_rect = False
    rect = None
    start_pos = None

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.2)

    # Try to position window
    try:
        manager = plt.get_current_fig_manager()
        if hasattr(manager, 'window') and hasattr(manager.window, 'wm_geometry'):
            manager.window.wm_geometry("+100+50")
    except Exception:
        pass  # Skip if window positioning is not supported

    ax.imshow(image)
    mode = ["points"]  # Using list to allow modification in nested functions

    def update_title():
        title = "Mode: Points - Click to mark points" if mode[
                                                             0] == "points" else "Mode: Box - Click and drag to draw box"
        ax.set_title(title)
        fig.canvas.draw_idle()

    def change_mode(label):
        mode[0] = label
        nonlocal rect, start_pos
        if rect:
            rect.remove()
            rect = None
        start_pos = None
        update_title()

    def on_mouse_press(event):
        if event.inaxes != ax:
            return
        nonlocal drawing_rect, start_pos, rect

        if mode[0] == "points":
            x, y = int(event.xdata), int(event.ydata)
            click_coords.append([x, y])
            ax.plot(x, y, 'r*', markersize=10)
            fig.canvas.draw()
        else:  # box mode
            drawing_rect = True
            start_pos = (event.xdata, event.ydata)
            print("on_mouse_press start_pos : ", start_pos)
            if rect:
                rect.remove()
            rect = plt.Rectangle(start_pos, 0, 0, edgecolor='green', facecolor='none', lw=2)
            ax.add_patch(rect)

    def on_mouse_move(event):
        if not drawing_rect or event.inaxes != ax or not start_pos:
            return

        if mode[0] == "box":
            w = event.xdata - start_pos[0]
            h = event.ydata - start_pos[1]
            rect.set_width(w)
            rect.set_height(h)
            fig.canvas.draw_idle()

    def on_mouse_release(event):
        nonlocal drawing_rect, box_coords
        if not drawing_rect or event.inaxes != ax or not start_pos:
            return

        if mode[0] == "box":
            drawing_rect = False
            # Récupérer les dimensions de l'image
            height, width = ax.get_images()[0].get_array().shape[:2]
            print("on_mouse_release start_pos : ", start_pos)
            print("on_mouse_release event.xdata : ", event.xdata)
            print("on_mouse_release event.ydata : ", event.ydata)
            # Convertir les coordonnées normalisées en pixels
            x0 = int(start_pos[0])
            y0 = int(start_pos[1])
            x1 = int(event.xdata)
            y1 = int(event.ydata)

            box_coords = [
                min(x0, x1),
                min(y0, y1),
                max(x0, x1),
                max(y0, y1)
            ]

    def on_finish(_):
        plt.close(fig)

    # Setup controls
    radio_ax = plt.axes([0.2, 0.05, 0.3, 0.1])
    radio = RadioButtons(radio_ax, ('points', 'box'))
    radio.on_clicked(change_mode)

    button_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
    btn_finish = Button(button_ax, 'Finish')
    btn_finish.on_clicked(on_finish)

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)

    update_title()
    plt.show(block=True)

    # Return results based on mode
    if mode[0] == "points" and click_coords:
        return np.array(click_coords), image_path, image, image_RGB, "points"
    elif mode[0] == "box" and box_coords:
        return np.array(box_coords), image_path, image, image_RGB, "box"
    else:
        print("No selection recorded.")
        return np.array([]), image_path, image, image_RGB, None


if __name__ == "__main__":
    coords, image_path, image, image_RGB, selection_type = capture_clicks_on_image()

    if coords.size > 0:
        predictor = load_sam2_model()
        predictor.set_image(image_RGB)

        if selection_type == "points":
            input_point = coords
            input_label = np.ones(len(input_point))
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

        elif selection_type == "box":
            print("coords: ", coords)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=coords[None, :],
                multimask_output=True,
            )
            show_masks(image, masks, scores, box_coords=coords)