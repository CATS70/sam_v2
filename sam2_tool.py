# Imports supplémentaires pour Google Drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials

# Imports pour SAM2
import torch
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os

# Interface Gradio pour l'application
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import gradio as gr

class Catalog:
    def __init__(self):
        self.objects = {}
        self.next_id = 0
        self.ensure_catalog_dir()
    
    def ensure_catalog_dir(self):
        if not os.path.exists("parasite_catalog"):
            os.makedirs("parasite_catalog")
    
    def add_object(self, image_path, mask, label, bbox):
        object_id = self.next_id
        self.objects[object_id] = {
            "image_path": image_path,
            "mask": mask,
            "label": label,
            "bbox": bbox
        }
        self.next_id += 1
        return object_id
    
    def get_catalog_summary(self):
        labels = {}
        for obj in self.objects.values():
            label = obj["label"]
            labels[label] = labels.get(label, 0) + 1
        return {
            "total_objects": len(self.objects),
            "labels": labels
        }

# Ajouter l'instance du catalogue au début de create_segmentation_app
catalog = Catalog()

def cleanup_temp_files():
    """Nettoie les fichiers temporaires"""
    import glob
    for f in glob.glob("temp_*"):
        try:
            os.remove(f)
        except:
            pass

# Interface Gradio pour l'application
def create_segmentation_app():
    # Ajouter ces variables comme attributs de l'application
    class AppState:
        def __init__(self):
            self.current_image_path = None
            self.current_image = None
            self.current_masks = None
            self.current_scores = None
            self.selected_mask_idx = 0
            self.current_folder_id = 'root'
            self.catalog = Catalog()
            self.drive = None
            self.predictor = None

    state = AppState()
    
    # Fonction d'initialisation de SAM2
    def init_sam2_model():
        """Initialise le modèle SAM2"""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Utilisation du périphérique: {device}")
            
            # Initialiser le modèle depuis Hugging Face
            MODEL_ID = "facebook/sam2-hiera-base-plus"
            predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID, device=str(device))
            
            print("Modèle SAM2 initialisé avec succès")
            return predictor
        except Exception as e:
            print(f"Erreur lors de l'initialisation du modèle SAM2: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    # Fonction d'initialisation de Google Drive
    def init_google_drive():
        """Initialise la connexion à Google Drive"""
        try:
            gauth = GoogleAuth()
            # Configuration des paramètres par défaut si les fichiers n'existent pas
            if not os.path.exists('client_secrets.json'):
                print("Attention: client_secrets.json manquant. Utilisation des paramètres par défaut.")
                gauth.DEFAULT_SETTINGS['client_config_file'] = 'client_secrets.json'
                
            if not os.path.exists('mycreds.txt'):
                print("Attention: mycreds.txt manquant. Une nouvelle authentification sera nécessaire.")
            
            # Tente d'utiliser les credentials existants
            gauth.LoadCredentialsFile("mycreds.txt")
            if gauth.credentials is None:
                # Authentification via le navigateur
                gauth.LocalWebserverAuth()
            elif gauth.access_token_expired:
                # Rafraîchit les credentials
                gauth.Refresh()
            else:
                # Initialise les credentials
                gauth.Authorize()
            # Sauvegarde les credentials
            gauth.SaveCredentialsFile("mycreds.txt")
            
            drive = GoogleDrive(gauth)
            print("Connexion à Google Drive établie avec succès")
            return drive
        except Exception as e:
            print(f"Erreur lors de la connexion à Google Drive: {str(e)}")
            return None

    # Fonction pour explorer les dossiers Google Drive
    def explore_drive_folders(drive, parent_id='root'):
        """
        Explore les dossiers dans Google Drive et renvoie une structure d'arborescence
        """
        file_list = drive.ListFile({
            'q': f"'{parent_id}' in parents and trashed=false"
        }).GetList()
        
        result = {"folders": [], "files": []}
        
        for file1 in file_list:
            if file1['mimeType'] == 'application/vnd.google-apps.folder':
                result["folders"].append({
                    "id": file1['id'],
                    "name": file1['title'],
                    "path": file1['id']
                })
            elif any(file1['title'].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                result["files"].append({
                    "id": file1['id'],
                    "name": file1['title'],
                    "path": file1['id']
                })
        
        return result

    # Fonction pour naviguer dans les dossiers Google Drive
    def load_drive_folders():
        nonlocal state
        if state.drive is None:
            return "Erreur: Impossible de se connecter à Google Drive"
        contents = explore_drive_folders(state.drive, state.current_folder_id)
        return "Structure de Google Drive chargée."

    # Fonction pour afficher les sous-dossiers et fichiers du dossier actuel
    def get_folder_contents(folder_id):
        nonlocal state
        
        if folder_id == "..":
            # Obtenir le parent du dossier actuel
            file_info = state.drive.CreateFile({'id': state.current_folder_id})
            file_info.FetchMetadata(fields='parents')
            if 'parents' in file_info and file_info['parents']:
                state.current_folder_id = file_info['parents'][0]['id']
            else:
                state.current_folder_id = 'root'
        else:
            state.current_folder_id = folder_id

        return explore_drive_folders(state.drive, state.current_folder_id)

    # Fonction pour sélectionner un dossier
    def select_folder(folder_id):
        contents = get_folder_contents(folder_id)
        folder_names = [f"{folder['name']} (dossier)" for folder in contents["folders"]]
        file_names = [f"{file['name']} (fichier)" for file in contents["files"]]

        return gr.Dropdown.update(choices=folder_names + file_names,
                                  value=None,
                                  label=f"Contenu de {folder_id}")

    # Fonction pour sélectionner une image
    def select_item(item_name, folder_contents):
        nonlocal state
        
        # Trouver l'item sélectionné
        selected_item = next(
            (item for item in folder_contents["files"] + folder_contents["folders"] 
             if item["name"] == item_name.split(" (")[0]),
            None
        )
        
        if selected_item is None:
            return None, "Élément non trouvé"

        if "(dossier)" in item_name:
            contents = get_folder_contents(selected_item["id"])
            return contents, None, "Navigation vers le dossier"
        else:
            # Télécharger l'image depuis Drive
            file1 = state.drive.CreateFile({'id': selected_item["id"]})
            temp_path = f"temp_{selected_item['name']}"
            file1.GetContentFile(temp_path)
            
            # Traiter l'image
            state.current_image_path = temp_path
            state.current_image = process_image_with_sam2(temp_path)
            
            # Afficher l'image
            plt.figure(figsize=(10, 10))
            plt.imshow(state.current_image)
            plt.axis('off')
            plt.tight_layout()

            # Convertir le plot en image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            data = base64.b64encode(buf.read()).decode('ascii')
            plt.close()

            return gr.Dropdown.update(), f"data:image/png;base64,{data}", f"Image chargée: {selected_item['name']}. Cliquez sur l'image pour sélectionner les objets parasites."

    def upload_image(image_file):
        nonlocal state

        # Sauvegarder l'image téléchargée
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_file)

        # Traiter l'image avec SAM2
        state.current_image_path = image_path
        state.current_image = process_image_with_sam2(image_path)

        # Afficher l'image
        plt.figure(figsize=(10, 10))
        plt.imshow(state.current_image)
        plt.axis('off')
        plt.tight_layout()

        # Convertir le plot en image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode('ascii')
        plt.close()

        return f"data:image/png;base64,{data}", "Image téléchargée avec succès. Cliquez sur l'image pour sélectionner les objets parasites."

    def segment_from_clicks(image_data, evt: gr.SelectData):
        nonlocal state

        if state.current_image is None:
            return image_data, "Veuillez d'abord télécharger une image."

        # Récupérer les coordonnées du clic
        x, y = evt.index
        points = [[x, y]]
        point_labels = [1]  # 1 pour foreground

        # Générer les masques
        masks, scores = generate_masks_from_points(state.current_image, points, point_labels)
        state.current_masks = masks
        state.current_scores = scores
        state.selected_mask_idx = 0  # Sélectionner le premier masque par défaut

        # Afficher l'image avec le masque
        plt.figure(figsize=(10, 10))
        plt.imshow(state.current_image)

        # Superposer le masque
        show_mask(masks[state.selected_mask_idx], plt.gca())
        show_points(points, point_labels, plt.gca())

        plt.title(f"Score du masque: {scores[state.selected_mask_idx]:.3f}")
        plt.axis('off')
        plt.tight_layout()

        # Convertir le plot en image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode('ascii')
        plt.close()

        # Préparer les options de masque
        mask_options = [f"Masque {i+1} (Score: {score:.3f})" for i, score in enumerate(scores)]

        return f"data:image/png;base64,{data}", f"Objet segmenté! Choisissez un masque et ajoutez-le au catalogue."

    def change_mask(mask_idx):
        nonlocal state

        if state.current_masks is None:
            return None, "Veuillez d'abord segmenter un objet."

        state.selected_mask_idx = mask_idx

        # Afficher l'image avec le masque sélectionné
        plt.figure(figsize=(10, 10))
        plt.imshow(state.current_image)

        # Superposer le masque
        show_mask(state.current_masks[state.selected_mask_idx], plt.gca())

        plt.title(f"Score du masque: {state.current_scores[state.selected_mask_idx]:.3f}")
        plt.axis('off')
        plt.tight_layout()

        # Convertir le plot en image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode('ascii')
        plt.close()

        return f"data:image/png;base64,{data}", f"Masque {state.selected_mask_idx+1} sélectionné."

    def add_to_catalog(label):
        nonlocal state

        if state.current_masks is None:
            return "Veuillez d'abord segmenter un objet."

        if not label:
            return "Veuillez entrer une étiquette pour l'objet parasite."

        # Récupérer le masque sélectionné
        mask = state.current_masks[state.selected_mask_idx]

        # Calculer la boîte englobante
        y_indices, x_indices = np.where(mask)
        x1, x2 = np.min(x_indices), np.max(x_indices)
        y1, y2 = np.min(y_indices), np.max(y_indices)
        bbox = [int(x1), int(y1), int(x2), int(y2)]

        # Ajouter au catalogue
        object_id = state.catalog.add_object(state.current_image_path, mask, label, bbox)

        # Récupérer le résumé du catalogue
        summary = state.catalog.get_catalog_summary()

        return f"Objet ajouté au catalogue avec ID: {object_id}\n\nRésumé du catalogue:\n- Total d'objets: {summary['total_objects']}\n- Étiquettes: {', '.join([f'{k} ({v})' for k, v in summary['labels'].items()])}"

    def export_catalog():
        import shutil
        # Créer un zip du catalogue localement
        shutil.make_archive("catalog", 'zip', "parasite_catalog")
        return "Catalogue exporté avec succès sous forme de fichier ZIP 'catalog.zip'."

    # Fonctions d'aide pour visualiser les masques et points
    def show_mask(mask, ax):
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    # Création de l'interface Gradio
    with gr.Blocks() as app:
        gr.Markdown("# Application de segmentation d'objets parasites avec SAM2")
        gr.Markdown("#### Utilise le modèle Meta Segment Anything 2 pour détecter et cataloguer les objets parasites dans les images d'empreintes")
        gr.Markdown("Cette application vous permet de sélectionner des objets parasites dans des images d'empreintes et de créer un catalogue pour entraîner un modèle de détection.")

        with gr.Row():
            with gr.Column(scale=1):
                # Panneau de navigation dans Google Drive
                load_drive_btn = gr.Button("Charger les dossiers")
                current_path_display = gr.Textbox(label="Chemin actuel", value=".")
                folder_browser = gr.Dropdown(label="Contenu du dossier", choices=[], interactive=True)

                # Alternative: téléchargement direct
                gr.Markdown("### Ou téléchargez directement une image:")
                upload_btn = gr.File(label="Télécharger une image")

            with gr.Column(scale=2):
                # Affichage et manipulation de l'image
                image_display = gr.Image(label="Image", interactive=True)
                status = gr.Textbox(label="Statut", value="Sélectionnez une image pour commencer.")

                with gr.Row():
                    mask_selector = gr.Slider(minimum=0, maximum=2, step=1, value=0, label="Sélectionner un masque", interactive=True)

                with gr.Row():
                    label_input = gr.Textbox(label="Étiquette de l'objet parasite")
                    add_btn = gr.Button("Ajouter au catalogue")

                catalog_status = gr.Textbox(label="Statut du catalogue", value="Aucun objet dans le catalogue.", lines=5)
                export_btn = gr.Button("Exporter le catalogue")

        # Variables pour stocker temporairement les données du navigateur de fichiers
        folder_contents = gr.State([])

        # Événements
        load_drive_btn.click(load_drive_folders, inputs=[], outputs=[status])
        load_drive_btn.click(lambda: get_folder_contents("."),
                            inputs=[],
                            outputs=[current_path_display, folder_contents, folder_contents])
        load_drive_btn.click(lambda x: [f"{folder['name']} (dossier)" for folder in x] + [f"{file['name']} (fichier)" for file in x],
                            inputs=[folder_contents],
                            outputs=[folder_browser])

        folder_browser.change(select_item,
                             inputs=[folder_browser, folder_contents],
                             outputs=[folder_browser, image_display, status])

        upload_btn.upload(upload_image, inputs=[upload_btn], outputs=[image_display, status])
        image_display.select(segment_from_clicks, inputs=[image_display], outputs=[image_display, status])
        mask_selector.change(change_mask, inputs=[mask_selector], outputs=[image_display, status])
        add_btn.click(add_to_catalog, inputs=[label_input], outputs=[catalog_status])
        export_btn.click(export_catalog, inputs=[], outputs=[catalog_status])
    return app

# Lancer l'application
app = create_segmentation_app()
app.launch(debug=True)

# Instructions d'utilisation
print("""
Instructions d'utilisation:
1. Lancez l'application
2. Cliquez sur 'Charger les dossiers' pour accéder à vos images locales
3. Naviguez dans la structure de vos dossiers et sélectionnez une image d'empreinte
4. Cliquez sur un objet parasite dans l'image pour le segmenter avec SAM
5. Utilisez le curseur pour sélectionner le meilleur masque parmi les options
6. Entrez une étiquette pour l'objet parasite (ex: 'poussière', 'cheveu', etc.)
7. Cliquez sur 'Ajouter au catalogue' pour sauvegarder l'objet
8. Répétez pour tous les objets parasites dans l'image
9. Utilisez 'Exporter le catalogue' pour créer une archive ZIP de votre catalogue

Note: Le catalogue sera enregistré dans le dossier 'parasite_catalog' de votre répertoire courant.
Ce catalogue pourra être utilisé ultérieurement pour entraîner votre propre modèle de détection.
""")

def process_image_with_sam2(image_path):
    """Traite l'image avec SAM2"""
    try:
        image = plt.imread(image_path)
        return image
    except Exception as e:
        print(f"Erreur lors du traitement de l'image: {str(e)}")
        return None

def generate_masks_from_points(image, points, point_labels):
    """Génère des masques à partir des points"""
    try:
        state.predictor.set_image(image)
        masks, scores, logits = state.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(point_labels)
        )
        return masks, scores
    except Exception as e:
        print(f"Erreur lors de la génération des masques: {str(e)}")
        return None, None

# Ajouter au début de create_segmentation_app
cleanup_temp_files()