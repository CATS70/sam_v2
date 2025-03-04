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
import webbrowser

# Ajouter ces constantes au début du fichier, après les imports
GOOGLE_DRIVE_MODEL_FOLDER = "SAM2_Models"  # Nom du dossier pour les modèles sur Drive
MODEL_FILENAME = "sam2_b.pt"  # Nom du fichier modèle sur Drive
INITIAL_DRIVE_FOLDER = "My Drive/MSPR/empreintes"  # Chemin complet par défaut

def init_sam2_model():
    """Initialise le modèle SAM2"""
    try:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Utiliser le modèle local
        model_path = "sam2_b.pt"  # Chemin vers votre fichier .pt
        if not os.path.exists(model_path):
            raise Exception(f"Modèle non trouvé : {model_path}")
            
        print(f"Chargement du modèle SAM2 depuis {model_path}")
        model = build_sam2_hf(model_id=model_path)
        predictor = SAM2ImagePredictor(model)
        print("Modèle SAM2 chargé avec succès")
        return predictor
    except Exception as e:
        print(f"Erreur lors de l'initialisation du modèle SAM2: {str(e)}")
        return None

def generate_masks_from_points(image, points, point_labels):
    """Génère des masques à partir des points"""
    try:
        predictor = init_sam2_model()
        if predictor is None:
            raise Exception("Impossible d'initialiser le modèle SAM2")
        
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(point_labels)
        )
        return masks, scores
    except Exception as e:
        print(f"Erreur lors de la génération des masques: {str(e)}")
        return None, None

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

# Déplacer cette fonction au début du fichier, juste après les imports
def process_image_with_sam2(image_path):
    """Traite l'image avec SAM2"""
    try:
        image = plt.imread(image_path)
        return image
    except Exception as e:
        print(f"Erreur lors du traitement de l'image: {str(e)}")
        return None

def load_drive_folders(state):
    """Charge le contenu du dossier initial sur Google Drive"""
    try:
        if state.drive is None:
            state.drive = state.init_google_drive()
            if state.drive is None:
                return (
                    "Non connecté",
                    [],
                    "Erreur: Configuration Google Drive manquante"
                )
        
        print("Recherche du dossier MSPR...")
        file_list = state.drive.ListFile({'q': "title='MSPR' and mimeType='application/vnd.google-apps.folder' and 'root' in parents and trashed=false"}).GetList()
        
        if not file_list:
            print("Dossier MSPR non trouvé")
            return (
                INITIAL_DRIVE_FOLDER,
                [],
                "Erreur: Dossier MSPR non trouvé dans My Drive"
            )
        
        mspr_folder = file_list[0]
        print(f"Dossier MSPR trouvé avec ID: {mspr_folder['id']}")
        
        print("Recherche du dossier empreintes...")
        file_list = state.drive.ListFile({'q': f"title='empreintes' and mimeType='application/vnd.google-apps.folder' and '{mspr_folder['id']}' in parents and trashed=false"}).GetList()
        
        if not file_list:
            print("Dossier empreintes non trouvé")
            return (
                INITIAL_DRIVE_FOLDER,
                [],
                "Erreur: Dossier empreintes non trouvé dans MSPR"
            )
        
        empreintes_folder = file_list[0]
        state.current_folder_id = empreintes_folder['id']
        print(f"Dossier empreintes trouvé avec ID: {state.current_folder_id}")
        
        print("Listage du contenu du dossier empreintes...")
        file_list = state.drive.ListFile({'q': f"'{state.current_folder_id}' in parents and trashed=false"}).GetList()
        
        folders = [f for f in file_list if f['mimeType'] == 'application/vnd.google-apps.folder']
        files = [f for f in file_list if f['mimeType'].startswith('image/')]
        
        print(f"Trouvé {len(folders)} dossiers et {len(files)} fichiers")
        
        choices = []
        for folder in sorted(folders, key=lambda x: x['title'].lower()):
            choice = f"{folder['title']} (dossier)"
            print(f"Ajout du dossier: {choice}")
            choices.append(choice)
            
        for file in sorted(files, key=lambda x: x['title'].lower()):
            choice = f"{file['title']} (fichier)"
            print(f"Ajout du fichier: {choice}")
            choices.append(choice)
        
        print(f"Liste finale des choix: {choices}")
        
        return (
            "My Drive/MSPR/empreintes",
            choices,
            f"Dossier My Drive/MSPR/empreintes chargé avec succès ({len(folders)} dossiers, {len(files)} fichiers)"
        )
        
    except Exception as e:
        print(f"Erreur lors du chargement des dossiers: {str(e)}")
        return INITIAL_DRIVE_FOLDER, [], f"Erreur: {str(e)}"

# Interface Gradio pour l'application
def create_segmentation_app():
    # Créer l'état de l'application
    class AppState:
        def __init__(self):
            self.current_image_path = None
            self.current_image = None
            self.current_masks = None
            self.current_scores = None
            self.selected_mask_idx = 0
            self.current_folder_id = None
            self.catalog = Catalog()
            self.drive = None
            self.predictor = None

        def init_google_drive(self):
            """Initialise la connexion à Google Drive"""
            try:
                gauth = GoogleAuth()
                gauth.LocalWebserverAuth()
                self.drive = GoogleDrive(gauth)
                print("Connexion à Google Drive établie avec succès")
                return self.drive
            except Exception as e:
                print(f"Erreur lors de la connexion à Google Drive: {str(e)}")
                return None

    state = AppState()
    
    # Définir les fonctions qui utilisent state
    def select_item_wrapper(item_name):
        """Wrapper pour select_item qui inclut state"""
        print(f"Sélection de : {item_name}")  # Debug
        if not item_name:
            return [], None, "Aucun élément sélectionné"
        
        try:
            if " (" not in item_name:
                return [], None, "Format de sélection invalide"
                
            name = item_name.split(" (")[0]
            is_folder = "(dossier)" in item_name
            
            print(f"Recherche de {name} ({'dossier' if is_folder else 'fichier'})")  # Debug
            
            # Chercher l'item dans les contenus actuels
            file_list = state.drive.ListFile({'q': f"'{state.current_folder_id}' in parents and trashed=false"}).GetList()
            
            if is_folder:
                folder = next((f for f in file_list if f['title'] == name and f['mimeType'] == 'application/vnd.google-apps.folder'), None)
                if folder:
                    state.current_folder_id = folder['id']
                    new_list = state.drive.ListFile({'q': f"'{state.current_folder_id}' in parents and trashed=false"}).GetList()
                    
                    # Séparer et trier les dossiers et fichiers
                    folders = sorted(
                        [f for f in new_list if f['mimeType'] == 'application/vnd.google-apps.folder'],
                        key=lambda x: x['title'].lower()
                    )
                    files = sorted(
                        [f for f in new_list if f['mimeType'].startswith('image/')],
                        key=lambda x: x['title'].lower()
                    )
                    
                    # Créer la liste des choix
                    choices = []
                    for f in folders:
                        choices.append(f"{f['title']} (dossier)")
                    for f in files:
                        choices.append(f"{f['title']} (fichier)")
                    
                    return choices, None, f"Dossier '{name}' ouvert ({len(folders)} dossiers, {len(files)} fichiers)"
            else:
                file = next((f for f in file_list if f['title'] == name and f['mimeType'].startswith('image/')), None)
                if file:
                    # Télécharger et afficher l'image
                    temp_path = f"temp_{file['title']}"
                    file.GetContentFile(temp_path)
                    
                    # Traiter l'image
                    state.current_image_path = temp_path
                    state.current_image = process_image_with_sam2(temp_path)
                    
                    if state.current_image is not None:
                        return None, state.current_image, f"Image '{name}' chargée"
                    else:
                        return None, None, f"Erreur lors du chargement de l'image '{name}'"
            
            return [], None, f"Élément '{name}' non trouvé"
            
        except Exception as e:
            print(f"Erreur lors de la sélection de l'item: {str(e)}")
            return [], None, f"Erreur: {str(e)}"

    def load_folders_wrapper():
        """Wrapper pour load_drive_folders qui inclut state"""
        return load_drive_folders(state)

    def segment_from_clicks_wrapper(image, evt: gr.SelectData):
        """Wrapper pour segment_from_clicks qui inclut state"""
        try:
            if image is None:
                return None, "Veuillez d'abord charger une image."
            
            if evt is None:
                return None, "Clic invalide sur l'image."
                
            # Convertir les coordonnées du clic
            x, y = evt.index
            points = [[y, x]]  # Inverser x,y car numpy utilise (y,x)
            point_labels = [1]  # 1 pour point positif
            
            # Générer les masques
            masks, scores = generate_masks_from_points(image, points, point_labels)
            
            if masks is None:
                return None, "Erreur lors de la génération des masques."
            
            # Sauvegarder les masques et scores
            state.current_masks = masks
            state.current_scores = scores
            state.selected_mask_idx = 0
            
            # Afficher l'image avec le masque
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(masks[0], plt.gca())
            plt.title(f"Score du masque: {scores[0]:.3f}")
            plt.axis('off')
            plt.tight_layout()
            
            return image, f"Masque généré avec un score de {scores[0]:.3f}"
            
        except Exception as e:
            print(f"Erreur lors de la segmentation: {str(e)}")
            return None, f"Erreur: {str(e)}"

    def change_mask_wrapper(mask_idx):
        """Wrapper pour change_mask qui inclut state"""
        if state.current_masks is None:
            return None, "Veuillez d'abord segmenter un objet."
        
        try:
            state.selected_mask_idx = int(mask_idx)
            
            # Afficher l'image avec le nouveau masque
            plt.figure(figsize=(10, 10))
            plt.imshow(state.current_image)
            show_mask(state.current_masks[state.selected_mask_idx], plt.gca())
            plt.title(f"Score du masque: {state.current_scores[state.selected_mask_idx]:.3f}")
            plt.axis('off')
            plt.tight_layout()
            
            return state.current_image, f"Masque {state.selected_mask_idx + 1} sélectionné"
        
        except Exception as e:
            print(f"Erreur lors du changement de masque: {str(e)}")
            return None, f"Erreur: {str(e)}"

    def add_to_catalog_wrapper(label):
        """Wrapper pour add_to_catalog qui inclut state"""
        if state.current_masks is None:
            return "Veuillez d'abord segmenter un objet."
        
        if not label:
            return "Veuillez entrer une étiquette pour l'objet parasite."
        
        try:
            # Récupérer le masque sélectionné
            mask = state.current_masks[state.selected_mask_idx]
            
            # Calculer la boîte englobante
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                return "Erreur: Masque vide"
            
            bbox = [
                int(np.min(x_indices)),
                int(np.min(y_indices)),
                int(np.max(x_indices)),
                int(np.max(y_indices))
            ]
            
            # Ajouter au catalogue
            object_id = state.catalog.add_object(
                state.current_image_path,
                mask,
                label,
                bbox
            )
            
            # Récupérer le résumé du catalogue
            summary = state.catalog.get_catalog_summary()
            return (
                f"Objet ajouté au catalogue avec ID: {object_id}\n\n"
                f"Résumé du catalogue:\n"
                f"- Total d'objets: {summary['total_objects']}\n"
                f"- Étiquettes: {', '.join([f'{k} ({v})' for k, v in summary['labels'].items()])}"
            )
        
        except Exception as e:
            print(f"Erreur lors de l'ajout au catalogue: {str(e)}")
            return f"Erreur: {str(e)}"

    def export_catalog_wrapper():
        """Wrapper pour export_catalog qui inclut state"""
        try:
            if state.catalog.objects:
                import shutil
                # Créer un zip du catalogue
                shutil.make_archive("catalog", 'zip', "parasite_catalog")
                return "Catalogue exporté avec succès sous forme de fichier ZIP 'catalog.zip'"
            else:
                return "Le catalogue est vide. Rien à exporter."
        except Exception as e:
            print(f"Erreur lors de l'exportation du catalogue: {str(e)}")
            return f"Erreur lors de l'exportation: {str(e)}"

    def upload_image_wrapper(file_obj):
        """Gère le téléchargement d'une image"""
        if not file_obj:
            return None, "Aucun fichier sélectionné"
        
        try:
            # Sauvegarder l'image téléchargée dans un fichier temporaire
            temp_path = f"temp_upload_{os.path.basename(file_obj.name)}"
            
            # Pour les fichiers uploadés via Gradio
            if hasattr(file_obj, 'name'):
                # Copier directement le fichier
                import shutil
                shutil.copy2(file_obj.name, temp_path)
            else:
                # Pour les cas où file_obj est une chaîne
                with open(temp_path, "wb") as f:
                    f.write(file_obj.encode())
            
            # Traiter l'image avec SAM2
            image = process_image_with_sam2(temp_path)
            
            if image is None:
                return None, "Erreur lors du traitement de l'image"
            
            # Mettre à jour l'état
            state.current_image_path = temp_path
            state.current_image = image
            
            return image, "Image téléchargée avec succès. Cliquez sur l'image pour sélectionner les objets parasites."
        
        except Exception as e:
            print(f"Erreur lors du téléchargement de l'image: {str(e)}")
            return None, f"Erreur: {str(e)}"

    # Charger les dossiers au démarrage
    initial_path, initial_contents, status_msg = load_drive_folders(state)
    
    with gr.Blocks(css="""
        /* Style pour la liste déroulante */
        .wide-dropdown {
            width: 100% !important;
        }
        .wide-dropdown select {
            width: 100% !important;
            min-width: 500px !important;
        }
        /* Forcer l'affichage complet du texte */
        .wide-dropdown option {
            width: 100% !important;
            white-space: normal !important;
            overflow: visible !important;
        }
    """) as app:
        gr.Markdown("# Application de segmentation d'objets parasites avec SAM2")
        gr.Markdown("#### Utilise le modèle Meta Segment Anything 2 pour détecter et cataloguer les objets parasites dans les images d'empreintes")
        gr.Markdown("Cette application vous permet de sélectionner des objets parasites dans des images d'empreintes et de créer un catalogue pour entraîner un modèle de détection.")

        with gr.Row():
            with gr.Column(scale=1):
                # Navigation Drive
                load_drive_btn = gr.Button("Charger les dossiers", variant="primary")
                current_path = gr.Textbox(
                    label="Chemin actuel",
                    value=INITIAL_DRIVE_FOLDER,
                    interactive=False  # Le chemin n'est pas modifiable directement
                )
                folder_browser = gr.Radio(
                    label="Contenu du dossier",
                    choices=initial_contents,
                    value=None,
                    interactive=True,
                    container=True,
                    scale=3
                )
                
                # Upload direct
                gr.Markdown("### Ou téléchargez directement une image:")
                upload_btn = gr.File(label="Télécharger une image")
            
            with gr.Column(scale=2):
                # Zone principale
                image_display = gr.Image(label="Image", interactive=True)
                status = gr.Textbox(label="Statut", value="Cliquez sur 'Charger les dossiers' pour commencer.")
                
                with gr.Row():
                    mask_selector = gr.Slider(
                        minimum=0,
                        maximum=2,
                        step=1,
                        value=0,
                        label="Sélectionner un masque",
                        interactive=True
                    )
                
                with gr.Row():
                    label_input = gr.Textbox(label="Étiquette de l'objet parasite")
                    add_btn = gr.Button("Ajouter au catalogue", variant="secondary")
                
                catalog_status = gr.Textbox(
                    label="Statut du catalogue",
                    value="Aucun objet dans le catalogue.",
                    lines=5
                )
                export_btn = gr.Button("Exporter le catalogue", variant="secondary")

        # Événements
        load_drive_btn.click(
            fn=load_folders_wrapper,
            inputs=[],
            outputs=[current_path, folder_browser, status]
        )
        
        folder_browser.change(
            fn=select_item_wrapper,
            inputs=[folder_browser],
            outputs=[folder_browser, image_display, status],
            show_progress=True  # Ajouter une barre de progression
        )

        upload_btn.upload(
            fn=upload_image_wrapper,
            inputs=[upload_btn],
            outputs=[image_display, status]
        )
        image_display.select(
            fn=segment_from_clicks_wrapper,
            inputs=[image_display],  # L'événement select ajoute automatiquement les coordonnées
            outputs=[image_display, status]
        )
        mask_selector.change(change_mask_wrapper, inputs=[mask_selector], outputs=[image_display, status])
        add_btn.click(add_to_catalog_wrapper, inputs=[label_input], outputs=[catalog_status])
        export_btn.click(export_catalog_wrapper, inputs=[], outputs=[catalog_status])

        # Initialiser les composants avec les valeurs initiales
        current_path.value = initial_path
        folder_browser.choices = initial_contents
        status.value = status_msg

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

def show_mask(mask, ax):
    """Affiche un masque sur un axe matplotlib"""
    color = np.array([30/255, 144/255, 255/255, 0.6])  # Bleu dodger transparent
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)