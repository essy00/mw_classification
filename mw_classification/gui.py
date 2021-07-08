from PIL import ImageTk, Image
import cv2
import numpy as np
import tensorflow

import tkinter.filedialog
import tkinter as tk
import shutil
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = tensorflow.keras.models.load_model('model.h5')


def ai(image_path: str):
    """
    Predicts if the person in the image is man or woman.
    Then with a face detection prints the prediction.

    Args:
        image_path (str): The image path.

    Returns:
        str: The result ("Man" or "Woman").
        int: The prediction of the machine (0 <= pred <= 1)
        np.array: The image to show
    """
    img_size = 64

    ai_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ai_image = cv2.resize(ai_image, (img_size, img_size))
    ai_image = np.array(ai_image).reshape(-1, img_size, img_size, 1) / 255.0

    pred = model.predict(ai_image)
    result = "Man" if pred < 0.5 else "Woman"

    real_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    real_image = cv2.resize(real_image, (640, 640))
    gray = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(real_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(
            real_image,
            result,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 69),
            2
        )

    return result, pred, real_image


class Root(tk.Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("AI")
        self.minsize(640, 400)

        self.labelFrame = tk.LabelFrame(self, text="Open A File")
        self.labelFrame.grid(column=0, row=1, padx=20, pady=20)
        self.upload_btn()

        self.image_path_label = tk.Label(text="")
        self.image_path_label.grid(column=0, row=2)

        self.image = None

        self.image_label = tk.Label()
        self.image_label.grid()

    def upload_btn(self):
        """
        When it's clicked, calls the self.file_dialog function.
        """
        button = tk.Button(
            self.labelFrame,
            text="Browse A File",
            command=self.file_dialog
        )
        button.grid(column=1, row=1)

    def file_dialog(self):
        """
        Takes the path, predicts and shows the returned image.
        """
        real_path = tk.filedialog.askopenfilename(
            initialdir="/",
            title="Select A File",
            filetypes=(
                ("jpeg", "*.jpg"),
                ("png", "*.png")
            )
        )

        tmp_path = f"./{real_path.split('/')[-1]}"

        try:
            shutil.copyfile(
                real_path,
                tmp_path
            )
        except Exception:
            pass

        result, pred, real_image = ai(tmp_path)

        os.remove(tmp_path)

        self.image_path_label.configure(
            text=f'Uploaded: {real_path.split("/")[-1]}'
        )
        self.image = ImageTk.PhotoImage(Image.fromarray(real_image))
        self.image_label.configure(image=self.image)
        self.image_label.image = self.image


def main():
    root = Root()
    root.mainloop()


if __name__ == '__main__':
    main()
