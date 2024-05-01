import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('99.7.h5')

class_name_dict = {
    0: 'ALL',
    1: 'AML',
    2: 'CLL',
    3: 'CML',
    4: 'Healthy'
}

class ImageClassifierApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.configure(bg='#87CEEB')  # Set the background color to blue
        self.title("Leukemia Diagnosis")

        # Set background image to fit the whole screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.bg_image = Image.open("D:/Leukemia/static/background.jpg")  #Background image
        self.bg_photo = ImageTk.PhotoImage(self.bg_image.resize((screen_width, screen_height)))
        self.bg_label = tk.Label(self, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)


        self.heading_label = tk.Label(self, text="Leukemia Diagnosis", font=("Helvetica", 20, "bold"), bg="#20B2AA", fg="white", pady=10)
        self.heading_label.pack(fill=tk.X)

        self.file_path = tk.StringVar()

        self.file_picker_label = tk.Label(self, text="Select an image:", font=("Helvetica", 12), fg="#2E4053")
        self.file_picker_label.pack(pady=5)

        self.file_picker = tk.Entry(self, textvariable=self.file_path, width=40, font=("Helvetica", 10), fg="#2E4053")
        self.file_picker.pack(pady=5, padx=10, ipady=5)

        self.browse_button = tk.Button(self, text="Browse", command=self.browse_image, font=("Helvetica", 12), bg="#3CB371", fg="white")
        self.browse_button.pack(pady=10)

        self.classify_button = tk.Button(self, text="Classify", command=self.on_classify, font=("Helvetica", 14), bg="#20B2AA", fg="white", pady=10)
        self.classify_button.pack()

        self.result_label = tk.Label(self, text="Result: N/A", font=("Helvetica", 14, "bold"),  fg="#2E4053", pady=10)
        self.result_label.pack()

        self.image_label = tk.Label(self, text="Selected Image", font=("Helvetica", 12),  fg="#2E4053")
        self.image_label.pack(pady=5)

        self.image_canvas = tk.Canvas(self, width=299, height=299, bg="white", borderwidth=2, relief="solid")
        self.image_canvas.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;*.jpeg")])
        self.file_path.set(file_path)

    def on_classify(self):
        image_path = self.file_path.get()
        if image_path:
            try:
                img = image.load_img(image_path, target_size=(299, 299))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img /= 255.0  # Normalize the image
                prediction = model.predict(img)

                # Get the predicted class index
                predicted_class = np.argmax(prediction)

                # Get the corresponding class label
                predicted_label = class_name_dict.get(predicted_class, 'Unknown')

                # Update the result label
                result = f"Class: {predicted_label}\nAccuracy: {prediction[0][predicted_class] * 100:.2f}%"
                self.result_label.config(text=f"Result: {result}")

                # Update the image canvas with the selected image
                img = Image.open(image_path)
                img.thumbnail((224, 224))
                img = ImageTk.PhotoImage(img)
                self.image_canvas.config(width=img.width(), height=img.height())
                self.image_canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.image_canvas.image = img

            except Exception as e:
                self.result_label.config(text=f"Error: {str(e)}")

        else:
            self.result_label.config(text="Please select an image.")

if __name__ == '__main__':
    app = ImageClassifierApp()
    app.mainloop()
