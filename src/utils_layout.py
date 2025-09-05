import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def draw_layout(
    image: Image.Image,
    phrases: list[str],
    boxes: list[tuple[float, float, float, float]],
) -> Image.Image:
    """
    Draws bounding boxes and annotates them with phrases on an image.

    Args:
        image (Image.Image): The input PIL image.
        phrases (list[str]): A list of phrases to annotate.
        boxes (list[tuple[float, float, float, float]]): A list of bounding boxes
            in [xmin, ymin, xmax, ymax] format, normalized to [0, 1].

    Returns:
        Image.Image: The image with bounding boxes and annotations.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Scale font size and box width based on image size
    base_width = 512  # Reference width for scaling
    scale_factor = width / base_width
    font_size = max(10, int(30 * scale_factor))
    box_width = max(1, int(3 * scale_factor))
    text_offset = font_size

    # Generate distinct colors for each box
    # Using a colormap to generate more distinct colors
    num_colors = len(boxes)
    colors = [tuple(int(c * 255) for c in plt.get_cmap("viridis", num_colors)(i)[:3]) for i in range(num_colors)]

    try:
        font = ImageFont.truetype("fonts/TimesNewRoman.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for i, (box, phrase) in enumerate(zip(boxes, phrases)):
        xmin, ymin, xmax, ymax = box

        # Denormalize coordinates
        abs_xmin = xmin * width
        abs_ymin = ymin * height
        abs_xmax = xmax * width
        abs_ymax = ymax * height

        # Get color for the box
        color = colors[i]

        # Draw bounding box
        draw.rectangle([(abs_xmin, abs_ymin), (abs_xmax, abs_ymax)], outline=color, width=box_width)

        # Draw text
        text_position = (abs_xmin, abs_ymin - text_offset) if abs_ymin - text_offset > 0 else (abs_xmin, abs_ymin + 5)

        # Draw a small background for the text for better readability
        text_bbox = draw.textbbox(text_position, phrase, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text(text_position, phrase, fill="white", font=font)

    return image
