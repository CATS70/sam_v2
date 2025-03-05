import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

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
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

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
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        
def load_sam2_model():
    # Obtenir le chemin absolu du répertoire courant
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construire les chemins absolus vers les fichiers
    sam2_checkpoint = os.path.join(current_dir, "models", "sam2.1_hiera_large.pt")
    model_cfg = os.path.join(current_dir, "models", "sam2.1_hiera_l.yaml")
    
    # Vérifier si les fichiers existent
    if not os.path.exists(sam2_checkpoint):
        print(f"Erreur: Le fichier de checkpoint {sam2_checkpoint} n'existe pas.")
        print(f"Contenu du dossier models: {os.listdir(os.path.join(current_dir, 'models'))}")
        raise FileNotFoundError(f"Fichier de checkpoint non trouvé: {sam2_checkpoint}")
    
    if not os.path.exists(model_cfg):
        # Chercher le fichier yaml dans le dossier models
        yaml_files = [f for f in os.listdir(os.path.join(current_dir, 'models')) if f.endswith('.yaml')]
        print(f"Erreur: Le fichier de configuration {model_cfg} n'existe pas.")
        print(f"Fichiers YAML disponibles dans le dossier models: {yaml_files}")
        
        # Vérifier si le sous-dossier sam2.1 existe
        sam2_1_dir = os.path.join(current_dir, 'models', 'sam2.1')
        if os.path.exists(sam2_1_dir) and os.path.isdir(sam2_1_dir):
            yaml_files_in_subdir = [f for f in os.listdir(sam2_1_dir) if f.endswith('.yaml')]
            print(f"Fichiers YAML disponibles dans le dossier models/sam2.1: {yaml_files_in_subdir}")
            
            # Si des fichiers yaml sont trouvés dans le sous-dossier, utiliser le premier
            if yaml_files_in_subdir:
                model_cfg = os.path.join(sam2_1_dir, yaml_files_in_subdir[0])
                print(f"Utilisation du fichier de configuration trouvé: {model_cfg}")
            # Sinon, si des fichiers yaml sont trouvés dans le dossier principal, utiliser le premier
            elif yaml_files:
                model_cfg = os.path.join(current_dir, 'models', yaml_files[0])
                print(f"Utilisation du fichier de configuration trouvé: {model_cfg}")
            else:
                raise FileNotFoundError(f"Aucun fichier de configuration YAML trouvé dans {os.path.join(current_dir, 'models')}")
        else:
            # Si le sous-dossier n'existe pas mais qu'il y a des fichiers yaml dans le dossier principal
            if yaml_files:
                model_cfg = os.path.join(current_dir, 'models', yaml_files[0])
                print(f"Utilisation du fichier de configuration trouvé: {model_cfg}")
            else:
                raise FileNotFoundError(f"Aucun fichier de configuration YAML trouvé dans {os.path.join(current_dir, 'models')}")
    
    print(f"Chargement du modèle avec:")
    print(f"- Checkpoint: {sam2_checkpoint}")
    print(f"- Configuration: {model_cfg}")
    
    # Charger le modèle
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    return predictor

# Fonction principale pour capturer les clics sur une image
def capture_clicks_on_image():
    # Demander le chemin du fichier image
    image_path = input("Entrez le chemin complet de l'image: ")
    
    # Vérifier si le fichier existe
    if not os.path.exists(image_path):
        print(f"Erreur: Le fichier {image_path} n'existe pas.")
        return
    
    # Charger l'image
    try:
        image = plt.imread(image_path)
        image_pil = Image.open(image_path)
        image_RGB = np.array(image_pil.convert("RGB"))
        print(f"Image chargée avec succès: forme {image.shape}, type {image.dtype}")
    except Exception as e:
        print(f"Erreur lors du chargement de l'image: {str(e)}")
        return
    
    # Liste pour stocker les coordonnées des clics
    click_coords = []
    
    # Fonction de rappel pour les clics
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            click_coords.append([x, y])
            print(f"Clic enregistré à la position: ({x}, {y})")
            # Afficher un marqueur à l'emplacement du clic
            ax.plot(x, y, 'r*', markersize=10)
            fig.canvas.draw()
    
    # Fonction de rappel pour le bouton "Terminer"
    def on_finish(event):
        plt.close(fig)
    
    # Créer la figure et l'axe
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.get_current_fig_manager().window.wm_geometry("+100+50")  # Position x=100, y=50 du coin supérieur gauche

    plt.subplots_adjust(bottom=0.2)  # Faire de la place pour le bouton
    
    # Afficher l'image
    ax.imshow(image)
    ax.set_title("Cliquez sur l'image pour marquer les points d'intérêt")
    
    # Ajouter un bouton "Terminer"
    ax_button = plt.axes([0.7, 0.05, 0.2, 0.075])
    btn_finish = Button(ax_button, 'Terminer')
    btn_finish.on_clicked(on_finish)
    
    # Connecter la fonction de rappel pour les clics
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    # Afficher la figure et attendre les interactions
    plt.show()
    
    # Convertir la liste des coordonnées en tableau numpy
    if click_coords:
        coords_array = np.array(click_coords)
        print(f"Coordonnées des clics enregistrées: \n{coords_array}")
        return coords_array, image_path, image, image_RGB
    else:
        print("Aucun clic enregistré.")
        return np.array([]), image_path, image

# Si ce script est exécuté directement
if __name__ == "__main__":
    coords, image_path, image, image_RGB = capture_clicks_on_image()
    if coords.size > 0:
        print(f"Tableau numpy des coordonnées: \n{coords}")
        print(f"Nombre de points: {len(coords)}")
        
        input_point = coords
        input_label = np.ones(len(input_point), dtype=int)

        predictor = load_sam2_model()
        
        predictor.set_image(image_RGB)
        
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        
        #masks, scores, _ = predictor.predict(
        #    point_coords=input_point,
        #    point_labels=input_label,
        #    mask_input=mask_input[None, :, :],
        #    multimask_output=False,
        #)       
        print(masks.shape)
        print(scores.shape)
        
        show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)
        # Afficher l'image avec tous les points marqués
        #lt.figure(figsize=(10, 10))
        #plt.imshow(image)
        #plt.scatter(coords[:, 0], coords[:, 1], color='red', marker='*', s=200)
        #plt.title(f"Image: {os.path.basename(image_path)} avec {len(coords)} points")
        #plt.axis('off')
        #plt.show()

