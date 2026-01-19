import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import difflib
import time
import pyperclip
from PIL import Image, ImageDraw, ImageFont # New import for text

# --- Configuration ---
HOLD_DURATION = 3
RESET_THRESHOLD = 0.7

COMMON_WORDS = [
    "ABOUT", "ABOVE", "AFTER", "AGAIN", "ALWAYS", "ANSWER", "APPLE",
    "BECAUSE", "BEFORE", "BEGIN", "BELOW", "BETTER", "BLACK", "BLUE", "CAT",
    "CHANGE", "CLEAN", "CLOSE", "COLOR", "COME", "COMPUTER",
    "DANCE", "DAILY", "DANGER", "DARK", "DATE", "DESIGN",
    "EAGLE", "EARTH", "EAT", "EDUCATION", "EIGHT", "EITHER",
    "FAMILY", "FATHER", "FEEL", "FIND", "FRIEND", "FUTURE",
    "GAME", "GARDEN", "GIVE", "GLASS", "GO", "GOOD", "GOOGLE", "GREAT", "GREEN",
    "HAPPY", "HELLO", "HELP", "HERE", "HOME", "HOUSE", "HOW",
    "IGLOO", "IMAGE", "IMPORTANT", "INDIA", "INFO", "INTERNET",
    "JACKET", "JOKE", "JOURNEY", "JUMP", "JUST",
    "KEEP", "KEY", "KIND", "KITCHEN", "KNOW",
    "LANGUAGE", "LARGE", "LEARN", "LEFT", "LIFE", "LIGHT", "LOVE",
    "MACHINE", "MAKE", "MANY", "MONEY", "MORE", "MORNING", "MOTHER",
    "NAME", "NEAR", "NEVER", "NIGHT", "NOTHING", "NOW", "NUMBER",
    "OCEAN", "OFFICE", "OFTEN", "ONLY", "OPEN", "OTHER",
    "PAPER", "PEOPLE", "PHONE", "PICTURE", "PLACE", "PLEASE", "POWER", "PROJECT", "PYTHON",
    "QUALITY", "QUESTION", "QUICK", "QUIET", "QUITE",
    "RAIN", "READ", "REAL", "RIGHT", "RIVER", "RUN",
    "SCHOOL", "SCREEN", "SEARCH", "SEE", "SHARE", "SISTER", "SMALL", "SMART", "SMILE", "SOUND", "START", "STOP", "STUDY",
    "TABLE", "TALK", "TEACH", "TEAM", "THANK", "THAT", "THERE", "THINK", "TIME", "TODAY", "TOGETHER",
    "UNDER", "UNIQUE", "UNTIL", "UP", "USE", "USER",
    "VALUE", "VIDEO", "VIEW", "VOICE",
    "WALK", "WATCH", "WATER", "WAY", "WELCOME", "WELL", "WHAT", "WHERE", "WHITE", "WHO", "WITH", "WORK", "WORLD", "WRITE",
    "YEAR", "YES", "YESTERDAY", "YOU", "YOUNG", "YOUR",
    "ZEBRA", "ZERO", "ZONE",
]

# --- Font Loading Helper ---
def load_font(size):
    try:
        # Try loading Arial (Windows standard)
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        try:
            # Try loading DejaVuSans (Linux standard)
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except IOError:
            # Fallback
            return ImageFont.load_default()

# Pre-load fonts to save performance
font_large = load_font(40)  # For Typed Word
font_medium = load_font(24) # For Suggestions
font_small = load_font(18)  # For Instructions

def put_text_pil(img, text, position, font, color):
    """
    Draws text on an OpenCV image using PIL.
    """
    # Convert OpenCV image (BGR) to PIL image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Draw text
    draw.text(position, text, font=font, fill=color)

    # Convert back to OpenCV image (BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_best_suggestion(partial_word, word_list):
    if not partial_word: return ""
    partial_upper = partial_word.upper()
    starts_with = [w for w in word_list if w.startswith(partial_upper)]
    if starts_with: return min(starts_with, key=len)
    matches = difflib.get_close_matches(partial_upper, word_list, n=1, cutoff=0.6)
    return matches[0] if matches else ""

# --- Init ---
try:
    model = tf.keras.models.load_model('isl_model.h5')
    labels = np.load('label_mapping.npy', allow_pickle=True)
except:
    print("Error: Model files not found.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

current_word = ""
suggestion = ""
last_pred_char = None
hold_start_time = 0
progress_percent = 0
feedback_msg = ""
feedback_timer = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    left_hand = [0.0] * 42
    right_hand = [0.0] * 42
    live_char = None

    # 1. Detection
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_type = handedness.classification[0].label
            temp = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]
            # Simplification: flattening correctly
            temp = []
            for lm in hand_landmarks.landmark:
                temp.extend([lm.x, lm.y])

            if hand_type == 'Right': right_hand = temp
            elif hand_type == 'Left': left_hand = temp

        final_input = np.array(left_hand + right_hand).reshape(1, -1).astype('float32')
        prediction = model.predict(final_input, verbose=0)
        if np.max(prediction) > RESET_THRESHOLD:
            live_char = labels[np.argmax(prediction)]

    # 2. Timer Logic
    current_time = time.time()
    if live_char:
        if live_char == last_pred_char:
            elapsed = current_time - hold_start_time
            progress_percent = min(elapsed / HOLD_DURATION, 1.0)
            if elapsed >= HOLD_DURATION:
                current_word += live_char
                hold_start_time = current_time
                progress_percent = 0
        else:
            last_pred_char = live_char
            hold_start_time = current_time
            progress_percent = 0
    else:
        last_pred_char = None
        progress_percent = 0

    suggestion = get_best_suggestion(current_word, COMMON_WORDS)

    # 3. UI Drawing (Using Pillow for Smooth Text)

    # Background Bar
    cv2.rectangle(frame, (0, 0), (640, 140), (30, 30, 30), -1)

    # Convert to PIL format once per frame for all text drawing
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # A. Typed Text
    draw.text((20, 40), f"{current_word}", font=font_large, fill=(255, 255, 255))

    # B. Suggestion
    if suggestion and suggestion != current_word:
        # Calculate width of current word to place suggestion next to it
        text_bbox = draw.textbbox((0, 0), current_word, font=font_large)
        text_width = text_bbox[2] - text_bbox[0]

        draw.text((20 + text_width + 10, 50), f"({suggestion})", font=font_medium, fill=(0, 255, 255))
        draw.text((400, 30), "TAB: Copy Suggestion", font=font_small, fill=(0, 255, 255))

    if current_word:
         draw.text((400, 60), "ENTER: Copy Typed", font=font_small, fill=(200, 200, 200))

    # C. Feedback Message (Big Green Text)
    if current_time - feedback_timer < 2:
        draw.text((150, 300), feedback_msg, font=font_large, fill=(0, 255, 0))

    # Convert back to OpenCV
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # D. Progress Bar (Draw with OpenCV as it's a shape, not text)
    if live_char:
        # We can draw simple text with cv2 for "Detecting" or use PIL above.
        # Using PIL above is cleaner, let's add it to the PIL block next time,
        # but for now let's use CV2 for the dynamic bar to keep it simple.

        cv2.putText(frame, f"Detecting: {live_char}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        bar_width = int(600 * progress_percent)
        cv2.rectangle(frame, (20, 115), (620, 130), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 115), (20 + bar_width, 130), (0, 255, 0), -1)
    else:
        cv2.putText(frame, "Show Hands", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)

    cv2.imshow('ISL Smart Interface', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == 8: current_word = current_word[:-1]
    elif key == 9:
        final_word = suggestion if suggestion else current_word
        if final_word:
            pyperclip.copy(final_word)
            feedback_msg = f"Copied: {final_word}"
            feedback_timer = current_time
            current_word = ""
    elif key == 13:
        if current_word:
            pyperclip.copy(current_word)
            feedback_msg = f"Copied: {current_word}"
            feedback_timer = current_time
            current_word = ""

cap.release()
cv2.destroyAllWindows()
