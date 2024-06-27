from PIL import Image, ImageFilter, ImageEnhance
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import cv2


def load_model():
    global model
    model = tf.keras.models.load_model('model_filter.h5')
    
def preprocess_image(image_path):
    image = Image.open(image_path)
    
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if exif is not None and 274 in exif:
            orientation = exif[274]
            if orientation == 3:
                image = image.rotate(180, expand=True)
                print("Rotated image 180 degrees")
            elif orientation == 6:
                image = image.rotate(270, expand=True)
                print("Rotated image 270 degrees")
            elif orientation == 8:
                image = image.rotate(90, expand=True)
                print("Rotated image 90 degrees")
        
    return image

def threshold_binary(image):
    threshold_value = 105
    thresholded = image.point(lambda p: p > threshold_value and 255)
    return thresholded

def refine_foreground(image):
    opened = image.filter(ImageFilter.MedianFilter(size=3)) 
    return opened

def brighten_object(image):
    brightness_enhancer = ImageEnhance.Brightness(image)
    brightened = brightness_enhancer.enhance(2)
    return brightened

def resize_image(image, size=(500, 150)):
    resized = image.resize(size, Image.LANCZOS)
    return resized

def full_preprocess(image_path):
    image = preprocess_image(image_path)
    thresholded = threshold_binary(image)
    refined = refine_foreground(thresholded)
    brightened = brighten_object(refined)
    resized = resize_image(brightened)
    return resized

def merge_close_bounding_boxes(bounding_boxes, threshold=10):
    merged_boxes = []
    while bounding_boxes:
        x, y, w, h = bounding_boxes.pop(0)
        merged = False
        for i, (mx, my, mw, mh) in enumerate(merged_boxes):
            if abs(x - mx) < threshold and abs(y - my) < threshold:
                nx = min(x, mx)
                ny = min(y, my)
                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny
                merged_boxes[i] = (nx, ny, nw, nh)
                merged = True
                break
        if not merged:
            merged_boxes.append((x, y, w, h))
    return merged_boxes

def is_contained_within(box1, box2):
    """ Check if box1 is contained within box2 """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2)

splitted_image = []

def split_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [image]

    merged_boxes = merge_horizontal_bounding_boxes(contours)
    merged_boxes.sort(key=lambda box: box[1])

    split_images = []
    padding = 5
    desired_width, desired_height = 500, 150  # Adjust as needed

    for box in merged_boxes:
        x, y, w, h = box
        x_pad = max(x - padding, 0)
        y_pad = max(y - padding, 0)
        w_pad = min(w + 2 * padding, image.shape[1] - x_pad)
        h_pad = min(h + 2 * padding, image.shape[0] - y_pad)

        row_image = image[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        row_image_resized = cv2.resize(row_image, (desired_width, desired_height), interpolation=cv2.INTER_LANCZOS4)

        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        row_image_resized = cv2.filter2D(row_image_resized, -1, kernel)

        split_images.append(row_image_resized)

    return split_images

def merge_horizontal_bounding_boxes(contours):
    # Function to merge horizontally aligned bounding boxes
    merged_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        merged = False
        for i, (mx, my, mw, mh) in enumerate(merged_boxes):
            if abs(y - my) < 10:  # Adjust threshold for merging vertically close boxes
                nx = min(x, mx)
                ny = min(y, my)
                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny
                merged_boxes[i] = (nx, ny, nw, nh)
                merged = True
                break
        if not merged:
            merged_boxes.append((x, y, w, h))
    return merged_boxes

roi_images = []

def annotate_image(image):
    image = np.array(image)
    image = image[:, :, ::-1].copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    black_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 27))
    closed_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_boxes = image.copy()

    bounding_boxes = []

    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        x -= 3
        y -= 3
        w += 6
        h += 6

        is_contained = False
        for box in bounding_boxes:
            if is_contained_within((x, y, w, h), box):
                is_contained = True
                break

        if not is_contained:
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
            bounding_boxes.append((x, y, w, h))

    merged_bounding_boxes = merge_close_bounding_boxes(bounding_boxes)

    merged_bounding_boxes.sort(key=lambda box: box[0])

    for idx, (x, y, w, h) in enumerate(merged_bounding_boxes):
        roi = image[y:y+h, x:x+w]
        
        if not roi.size == 0:
            roi_images.append(roi)
        else:
            print(f'Skipping empty ROI: ')

    
    return roi_images

selected_images = []

def remove_small_roi_images(image, min_width=10, min_height=4):
    width, height = image_dimensions(image)
    if width < min_width or height < min_height:
        pass
    else:
        selected_images.append(image)

def image_dimensions(image):
    height, width, _ = image.shape
    return width, height

def add_white_background(image, canvas_size=(100, 100)):
    if image.shape[1] > canvas_size[0] or image.shape[0] > canvas_size[1]:
        scale = min(canvas_size[0] / image.shape[1], canvas_size[1] / image.shape[0])
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    
    x_offset = (canvas_size[0] - image.shape[1]) // 2
    y_offset = (canvas_size[1] - image.shape[0]) // 2
    
    canvas[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
    
    return canvas

def process_images_in_folder(image):
    image_with_background = add_white_background(image)
    resized_image = cv2.resize(image_with_background, (180, 180), interpolation=cv2.INTER_AREA)
    return resized_image

label_map = {}
def get_data(img):
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gaussian_3 = cv2.GaussianBlur(img, (9, 9), 10.0)
    img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    
    img = np.expand_dims(img, axis=0)
    
    return img

label_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '-', 12: '*', 13: '/', 14: '=',
    15: '(', 16: ')', 17: '{', 18: '}', 19: '[', 20: ']'
}


def evaluate_expression(expression):
    def precedence(op):
        if op in {'+', '-'}:
            return 1
        if op in {'*', '/'}:
            return 2
        return 0

    def apply_op(a, b, op):
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/':
            if b == 0:
                raise ValueError("Division by zero error")  # Added check for division by zero
            return a / b

    def is_opening_bracket(char):
        return char in '([{'

    def is_closing_bracket(char):
        return char in ')]}'

    def brackets_match(opening, closing):
        pairs = {'(': ')', '[': ']', '{': '}'}
        return pairs[opening] == closing

    def infix_to_postfix(tokens):
        output = []
        stack = []
        for token in tokens:
            if token.isdigit() or (token[0] == '-' and token[1:].isdigit()):
                output.append(token)
            elif is_opening_bracket(token):
                stack.append(token)
            elif is_closing_bracket(token):
                while stack and not is_opening_bracket(stack[-1]):
                    output.append(stack.pop())
                if stack and brackets_match(stack[-1], token):
                    stack.pop()  # Remove opening bracket
                else:
                    raise ValueError("Mismatched brackets")  # Changed return to raise
            else:
                while stack and not is_opening_bracket(stack[-1]) and precedence(stack[-1]) >= precedence(token):
                    output.append(stack.pop())
                stack.append(token)
        while stack:
            if is_opening_bracket(stack[-1]):
                raise ValueError("Mismatched brackets")  # Changed return to raise
            output.append(stack.pop())
        return output

    def evaluate_postfix(tokens):
        stack = []
        for token in tokens:
            if token.isdigit() or (token[0] == '-' and token[1:].isdigit()):
                stack.append(float(token))
            else:
                try:
                    b = stack.pop()
                    a = stack.pop()
                except IndexError as e:
                    raise ValueError(f"Error evaluating postfix expression: stack underflow. Token: {token}, Stack: {stack}") from e
                
                if token == '/' and b == 0:
                    raise ValueError("Division by zero error")
                stack.append(apply_op(a, b, token))
        if len(stack) != 1:
            raise ValueError(f"Error evaluating postfix expression: too many items left on stack. Stack: {stack}")
        return stack[0]


    tokens = []
    current_token = ''
    for char in expression:
        if char.isdigit() or (char == '-' and (not current_token or not current_token[-1].isdigit())):
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ''
            if char != ' ':
                tokens.append(char)
    if current_token:
        tokens.append(current_token)

    postfix = infix_to_postfix(tokens)
    return evaluate_postfix(postfix)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict(image_path):
    try:
        processed_image = full_preprocess(image_path)  
        processed_image = np.array(processed_image)
        split_images = split_image(processed_image)

        final_result = []  
        
        verdicts = []
        
        for img in split_images:
            annotated_image = annotate_image(img)
            selected_images = [
                ann_img for ann_img in annotated_image 
                if ann_img.shape[0] >= 4 and ann_img.shape[1] >= 10
            ]
            
            processed_selected_images = [process_images_in_folder(img) for img in selected_images]
            
            line_predictions = []
            
            for proc_img in processed_selected_images:
                img_data = get_data(proc_img)
                prediction = model.predict(img_data)
                predicted_class = np.argmax(prediction, axis=1)
                predicted_char = label_map[predicted_class[0]]
                
                line_predictions.append(predicted_char)
                
                # cleanup
                del img_data
                del prediction
                del predicted_class
                del predicted_char
            
            line_result = ''.join(line_predictions)
            
            def process_string(input_string):
                equals_position = input_string.find('=')
                
                if equals_position != -1:
                    result = input_string[equals_position+1:]
                    return result
                else:
                    return input_string
                
            line = process_string(line_result)
            
            try:
                verdict = evaluate_expression(line)
            except ValueError as e:
                print(f"Error evaluating expression '{line}': {e}")
                verdict = None
            
            verdicts.append(verdict)
            
            final_result.append(line_result)
            
            #cleanup
            del line_result
            processed_selected_images.clear()
            selected_images.clear()
            annotated_image.clear()

        # Cleanup
        del processed_image
        split_images.clear()
        
        
        verdict_map = {}
        
        for f, v in zip(final_result, verdicts):
            if verdicts[0] != v:
                verdict_map[f] = "Wrong"
            else:
                verdict_map[f] = "Right"

        # # Print or use verdict_map as needed
        # for key, value in verdict_map.items():
        #     print(f"Key: {key}, Value: {value}")
            

        
        return verdict_map

    except Exception as e:
        raise RuntimeError(f"Error predicting image: {str(e)}")


