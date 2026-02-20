from keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from keras.applications.vgg19 import VGG19, preprocess_input as vgg_preprocess
from PIL import Image
import tempfile
import os

# ----------------------------------------------------------------------
# Load Model and Labels
# ----------------------------------------------------------------------
MODEL_FILE = "models/my_model.hdf5"

dog_names = [
    "Affenpinscher", "Afghan hound", "Airedale terrier", "Akita", "Alaskan malamute",
    "American eskimo dog", "American foxhound", "American staffordshire terrier",
    "American water spaniel", "Anatolian shepherd dog", "Australian cattle dog",
    "Australian shepherd", "Australian terrier", "Basenji", "Basset hound", "Beagle",
    "Bearded collie", "Beauceron", "Bedlington terrier", "Belgian malinois",
    "Belgian sheepdog", "Belgian tervuren", "Bernese mountain dog", "Bichon frise",
    "Black and tan coonhound", "Black russian terrier", "Bloodhound", "Bluetick coonhound",
    "Border collie", "Border terrier", "Borzoi", "Boston terrier", "Bouvier des flandres",
    "Boxer", "Boykin spaniel", "Briard", "Brittany", "Brussels griffon", "Bull terrier",
    "Bulldog", "Bullmastiff", "Cairn terrier", "Canaan dog", "Cane corso",
    "Cardigan welsh corgi", "Cavalier king charles spaniel", "Chesapeake bay retriever",
    "Chihuahua", "Chinese crested", "Chinese shar-pei", "Chow chow", "Clumber spaniel",
    "Cocker spaniel", "Collie", "Curly-coated retriever", "Dachshund", "Dalmatian",
    "Dandie dinmont terrier", "Doberman pinscher", "Dogue de bordeaux",
    "English cocker spaniel", "English setter", "English springer spaniel",
    "English toy spaniel", "Entlebucher mountain dog", "Field spaniel", "Finnish spitz",
    "Flat-coated retriever", "French bulldog", "German pinscher", "German shepherd dog",
    "German shorthaired pointer", "German wirehaired pointer", "Giant schnauzer",
    "Glen of imaal terrier", "Golden retriever", "Gordon setter", "Great dane",
    "Great pyrenees", "Greater swiss mountain dog", "Greyhound", "Havanese",
    "Ibizan hound", "Icelandic sheepdog", "Irish red and white setter", "Irish setter",
    "Irish terrier", "Irish water spaniel", "Irish wolfhound", "Italian greyhound",
    "Japanese chin", "Keeshond", "Kerry blue terrier", "Komondor", "Kuvasz",
    "Labrador retriever", "Lakeland terrier", "Leonberger", "Lhasa apso", "Lowchen",
    "Maltese", "Manchester terrier", "Mastiff", "Miniature schnauzer", "Neapolitan mastiff",
    "Newfoundland", "Norfolk terrier", "Norwegian buhund", "Norwegian elkhound",
    "Norwegian lundehund", "Norwich terrier", "Nova scotia duck tolling retriever",
    "Old english sheepdog", "Otterhound", "Papillon", "Parson russell terrier",
    "Pekingese", "Pembroke welsh corgi", "Petit basset griffon vendeen", "Pharaoh hound",
    "Plott", "Pointer", "Pomeranian", "Poodle", "Portuguese water dog", "Saint bernard",
    "Silky terrier", "Smooth fox terrier", "Tibetan mastiff", "Welsh springer spaniel",
    "Wirehaired pointing griffon", "Xoloitzcuintli", "Yorkshire terrier"
]

# ----------------------------------------------------------------------
# Load Pretrained Models
# ----------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml")
ResNet50_model = ResNet50(weights="imagenet")
my_model = load_model(MODEL_FILE)

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def read_uploaded_image(uploaded_file):
    """Convert uploaded file to OpenCV format (BGR)."""
    image_pil = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image_pil)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

def path_to_tensor(img_path):
    """Convert image path to 4D tensor (1, 224, 224, 3)."""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def face_detector(img_path):
    """Return True if a face is detected in image."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    """Return predicted ImageNet class label."""
    img = preprocess_input(path_to_tensor(img_path))
    preds = ResNet50_model.predict(img)
    return np.argmax(preds)

def dog_detector(img_path):
    """Return True if a dog is detected."""
    prediction = ResNet50_predict_labels(img_path)
    return 151 <= prediction <= 268

def extract_VGG19(tensor):
    """Extract bottleneck features using VGG19."""
    model = VGG19(weights="imagenet", include_top=False)
    return model.predict(vgg_preprocess(tensor))

def my_predict_breed(img_path):
    """Predict dog breed."""
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    predicted_vector = my_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

def which_breed(img_path):
    """Detect face/dog and predict breed."""
    if face_detector(img_path):
        return "This person resembles a " + my_predict_breed(img_path)
    elif dog_detector(img_path):
        return "This dog is a " + my_predict_breed(img_path)
    else:
        return "Neither a dog nor a human face detected in this picture."

# ----------------------------------------------------------------------
# Web-App Safe Handler (Fixes imread temp file issue)
# ----------------------------------------------------------------------
def which_breed_from_upload(uploaded_file):
    """
    Handles Streamlit/Flask uploaded files safely.
    Reads file from memory → saves temporarily → runs detection.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img = read_uploaded_image(uploaded_file)
        cv2.imwrite(tmp.name, img)
        result = which_breed(tmp.name)
    os.remove(tmp.name)
    return result
