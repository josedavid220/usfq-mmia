import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def visualize_predictions(image, predictions, category_names, threshold=0.5):
    plt.figure(figsize=(12, 8))
    image = image.permute(1, 2, 0).numpy()
    image = (
        image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    ).clip(0, 1)

    plt.imshow(image)

    ax = plt.gca()

    for box, label, score in zip(
        predictions["boxes"], predictions["labels"], predictions["scores"]
    ):
        if score >= threshold:
            box = box.cpu().numpy()

            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

            plt.text(
                box[0],
                box[1],
                f"{category_names[label]}: {score:.2f}",
                color="red",
                fontsize=12,
            )

    plt.axis("off")
    plt.show()
