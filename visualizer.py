# _____________ imports _____________
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import pygame
import sys

from tensorflow.python.keras.models import load_model, Model

from model_generator import train

# _____________ constants/initializing _____________

# Load the model
model = tf.keras.models.load_model("model/digit_recognizer.keras")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND = (150, 150, 200)
LIGHT_GRAY = (200, 200, 200)

# _____________ classes _____________

class NodeVis:
    def __init__(self, activation):
        self.activation = activation
        self.color = (255-(activation*255), activation*255, 0)
    def draw(self, surface, center):
        pygame.draw.circle(surface, self.color, center, 15)

class Layer:
    def __init__(self, layer_num, output = False):
        self.layer_num = layer_num
        self.nodes = []
        self.output = output

    def add_node(self, node):
        self.nodes.append(node)

    def draw_layer(self):
        delta = (HEIGHT-100)/(self.nodes.__len__())

        for i in range(self.nodes.__len__()):
            x = self.layer_num * (WIDTH - 500) / layer_total + 400
            y = 100 + (delta * i)
            self.nodes[i].draw(screen, (x, y))

            if self.output:
                label_text = font.render(str(i), True, (0, 0, 0))
                label_pos = (x + 30, y - label_text.get_height() // 2)
                screen.blit(label_text, label_pos)

# Button Class
class Button:
    def __init__(self, x, y, width, height, text, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.color = LIGHT_GRAY

    def draw(self, surface):
        # Draw button rectangle
        pygame.draw.rect(surface, self.color, self.rect)
        # Draw text centered in button
        text_surf = font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_click(self, pos):
        # If mouse position is within the button rect, run callback
        if self.rect.collidepoint(pos):
            self.callback()

# Example callback function
def on_button_click():
    canvas_str = pygame.image.tostring(canvas_surface, "RGB")
    image = Image.frombytes("RGB", (280, 280), canvas_str)

    # Convert to grayscale and resize
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    image_arr = np.array(image) / 255.0
    image_arr = image_arr.reshape(1, 28, 28)

    # Load model
    model =  tf.keras.models.load_model("model/digit_recognizer.keras")

    model.predict(image_arr)

    inputs = tf.keras.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten()(inputs)
    x1 = tf.keras.layers.Dense(128, activation='relu')(x)
    x2 = tf.keras.layers.Dense(128, activation='relu')(x1)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x2)

    activation_model = tf.keras.Model(inputs=inputs, outputs=[x1, x2, outputs])

    # Load MNIST test image to test activations
    (_, _), (img_test, _) = tf.keras.datasets.mnist.load_data()
    img_test = tf.keras.utils.normalize(img_test, axis=1)

    input_image = img_test[0].reshape(1, 28, 28)  # Single input

    # Get activations
    activations = activation_model.predict(input_image)

    prediction = np.argmax(activations[len(activations)-1])

    print(len(layers))
    canvas_surface.fill(WHITE)

    for i in range(len(activations) - 1):  # Skip last if it's output layer
        layers[i].modify_layer(activations[i])



def retrain_model():
    train(layer_total, node_list)




def toggle_fullscreen():
    global is_fullscreen, screen
    is_fullscreen = not is_fullscreen
    if is_fullscreen:
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((800, 600))  # Go back to windowed mode


# UI Buttons
buttons = [
    Button(100, 750, 150, 50, "Predict Digit", on_button_click),
    Button(100, 850, 150, 50, "Retrain Model", retrain_model),
    Button(100, 950, 150, 50, "Exit", lambda: sys.exit())
]


# _____________ app logic _____________
print("<Welcome to the Interactive Neural Network>")
print("Please fill in all information before opening the application")
layer_total = int(input("Number of hidden layers: ")) + 1

node_list = []

for j in range(layer_total-1):
    nodes = int(input("Number of nodes in hidden layer #"+str(j+1)+": "))
    node_list.append(nodes)


# Initialize pygame
pygame.init()

# Creates the display
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
is_fullscreen = True

pygame.display.set_caption("Interactive Neural Network")

# Set up "in game" clock
clock = pygame.time.Clock()

# font used for buttons/text
font = pygame.font.SysFont("Arial", 24)

canvas_rect = pygame.Rect(50, 300, 280, 280)  # Your drawing box (10x scale of 28x28)
canvas_surface = pygame.Surface((280, 280))
canvas_surface.fill(WHITE)
drawing = False
#drawing board, so to speak, for the input digit

layers = []

for k in range(layer_total-1):
    layer = Layer(k+1)
    for l in range(node_list[k]):
        layer.add_node(NodeVis(0))
    layers.append(layer)

output = Layer(layer_total, True)
for i in range(10):
    output.add_node(node = NodeVis(0))
layers.append(output)

running = True

while running:
    screen.fill(BACKGROUND)  # Sets the background color to white


    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        #quits upon clicking the "quit" button

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
            # Makes 'F' toggle fullscreen
                toggle_fullscreen()
            elif event.key == pygame.K_ESCAPE:
                running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if canvas_rect.collidepoint(event.pos):
                drawing = True
            if event.button == 1:  # Clicking with the left mouse button
                for button in buttons:
                    button.check_click(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            if canvas_rect.collidepoint(event.pos):
                pygame.draw.circle(canvas_surface, BLACK, (event.pos[0] - canvas_rect.x, event.pos[1] - canvas_rect.y),
                                   8)

    # Draw the canvas area to the screen
    screen.blit(canvas_surface, canvas_rect.topleft)
    pygame.draw.rect(screen, BLACK, canvas_rect, 2)  # Outline

    # Draw all buttons
    for button in buttons:
        button.draw(screen)

    for layer in layers:
        layer.draw_layer()

    pygame.display.flip()  # Update display
    clock.tick(60)  # Limit to 60 FPS

pygame.quit()
sys.exit()