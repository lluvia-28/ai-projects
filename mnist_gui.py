import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DigitRecognitionGUI:
    def __init__(self):
        self.load_and_train_model()

        # create GUI
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("800x600")

        self.setup_drawing_area()
        self.setup_prediction_area()
        self.setup_buttons()

    def load_and_train_model(self):
        print("Loading MNIST data and training the model...")
        
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

        # Normalize
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.fit(train_images, train_labels, epochs=3, verbose=1)

        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=0)
        print(f"Model trained! Test accuracy: {test_acc:.4f}")

    def setup_drawing_area(self):
        draw_frame = ttk.Frame(self.root)
        draw_frame.pack(side=tk.LEFT, padx=10, pady=10)

        ttk.Label(draw_frame, text="Draw a digit (0-9):", font=("Arial", 14)).pack()

        # Canvas for drawing
        self.canvas = tk.Canvas(draw_frame, width=280, height=280, bg='black')
        self.canvas.pack(pady=5)

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        # PIL Image
        self.image = Image.new('L', (280, 280), 0)
        self.draw_on_image = ImageDraw.Draw(self.image)

    def setup_prediction_area(self):
        # Frame for predictions
        pred_frame = ttk.Frame(self.root)
        pred_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        ttk.Label(pred_frame, text='Prediction:', font=("Arial", 14)).pack()

        self.pred_label = ttk.Label(pred_frame, text="Draw a digit to predict",
                                    font=("Arial", 14))
        self.pred_label.pack(pady=5)

        self.confidence_label = ttk.Label(pred_frame, text="", font=("Arial", 14))
        self.confidence_label.pack()

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, pred_frame)
        self.canvas_plot.get_tk_widget().pack(pady=10)

        self.update_probability_plot([0] * 10)

    def setup_buttons(self):
        # Create control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, pady=20, fill=tk.X)

        # Center the buttons
        inner_frame = ttk.Frame(button_frame)
        inner_frame.pack(expand=True)

        predict_btn = ttk.Button(inner_frame, text="🔍 PREDICT", command=self.predict_digit, width=15)
        predict_btn.pack(side=tk.LEFT, padx=10)
        
        clear_btn = ttk.Button(inner_frame, text="🗑️ CLEAR", command=self.clear_canvas, width=15)
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        exit_btn = ttk.Button(inner_frame, text="❌ EXIT", command=self.root.quit, width=15)
        exit_btn.pack(side=tk.LEFT, padx=10)
    
    def draw(self, event):
        x, y = event.x, event.y
        r = 8  # brush radius
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        # draw on PIL image
        self.draw_on_image.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def clear_canvas(self):
        # clear the drawing canvas
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw_on_image = ImageDraw.Draw(self.image)
        self.pred_label.config(text="Draw a digit to predict")
        self.confidence_label.config(text="")
        self.update_probability_plot([0] * 10)

    def predict_digit(self):
        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
         
        # convert to numpy array and normalize
        img_array = np.array(resized) / 255.0

        img_array = img_array.reshape(1, 28, 28)

        # make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        # update display
        self.pred_label.config(text=f"Predicted: {predicted_digit}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")

        # update probability plot
        self.update_probability_plot(predictions[0])

    def update_probability_plot(self, probabilities):
        self.ax.clear()
        digits = list(range(10))
        bars = self.ax.bar(digits, probabilities, color='skyblue', alpha=0.7)

        # highlight the highest probability
        max_idx = np.argmax(probabilities)
        bars[max_idx].set_color('red')

        self.ax.set_xlabel('Digit')
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Prediction Probabilities')
        self.ax.set_xticks(digits)
        self.ax.set_ylim(0, 1)

        # add percentage labels on bars
        for i, (digit, prob) in enumerate(zip(digits, probabilities)):
            if prob > 0.01:  # only show labels for probs > 1%
                self.ax.text(digit, prob + 0.01, f'{prob*100:.1f}%',
                             ha='center', va='bottom', fontsize=8)
        
        self.canvas_plot.draw()

    def run(self):
        print("Starting GUI...")
        self.root.mainloop()

if __name__ == "__main__":
    app = DigitRecognitionGUI()
    app.run()





