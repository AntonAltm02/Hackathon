# from manim import *
from manimlib.imports import *
# Define a custom scene for the CNN visualization
class CNNVisualization(Scene):
    def construct(self):
        # Load and display an image (replace the path with your image file)
        image = ImageMobject("data/normal.png")
        image.scale(0.5)  # adjust size as needed
        image.to_edge(LEFT)
        self.play(FadeIn(image))
        self.wait(0.5)

        # Create an arrow to indicate the image being fed into the network
        arrow = Arrow(start=image.get_right(), end=image.get_right() + RIGHT * 2, buff=0.1)
        self.play(GrowArrow(arrow))
        self.wait(0.5)

        # Simulate the CNN block as a rectangle with a label
        cnn_box = Rectangle(width=2.5, height=2, color=BLUE)
        cnn_box.next_to(arrow.get_end(), RIGHT, buff=0.5)
        cnn_label = Text("CNN", font_size=36)
        cnn_label.move_to(cnn_box.get_center())
        self.play(FadeIn(cnn_box), Write(cnn_label))
        self.wait(0.5)

        # Create an arrow leaving the CNN block toward the outputs
        arrow_out = Arrow(start=cnn_box.get_right(), end=cnn_box.get_right() + RIGHT * 2, buff=0.1)
        self.play(GrowArrow(arrow_out))
        self.wait(0.5)

        # Prepare output class labels
        class_names = [
            "0: Negative",
            "1: Benign Calcification",
            "2: Benign Mass",
            "3: Malignant Calcification",
            "4: Malignant Mass"
        ]
        # Create Text mobjects for each class and arrange them vertically
        output_labels = VGroup(*[Text(label, font_size=28) for label in class_names])
        output_labels.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        output_labels.next_to(arrow_out.get_end(), RIGHT, buff=0.5)

        # Highlight the predicted output (for example, the second class)
        predicted_index = 1  # change index [0-4] for different output predictions
        predicted_rect = SurroundingRectangle(output_labels[predicted_index], color=YELLOW, buff=0.1)

        # Animate the output labels appearance.
        self.play(FadeIn(output_labels, shift=RIGHT))
        self.wait(0.5)
        self.play(Create(predicted_rect))
        self.wait(2)
