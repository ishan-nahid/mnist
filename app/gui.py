import tkinter as tk
from tkinter import filedialog
from model import predict

def create_gui():
    root = tk.Tk()
    root.title("Image Classifier")

    def select_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            prediction = predict(file_path)
            result_label.config(text=f"Prediction: {prediction}")

    upload_button = tk.Button(root, text="Upload Image", command=select_image)
    upload_button.pack(pady=10)

    result_label = tk.Label(root, text="")
    result_label.pack()

    return root