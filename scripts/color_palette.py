# color_palette.py
import matplotlib.pyplot as plt

PALETTE = {
    'Black': '#000000',
    'Orange': '#E69F00',
    'Sky Blue': '#56B4E9',
    'Bluish Green': '#009E73',
    'Yellow': '#F0E442',
    'Blue': '#0072B2',
    'Vermillion': '#D55E00',
    'Reddish Purple': '#CC79A7'
}

def hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color to RGBA."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3)) + (alpha,)

def get_color(color_name, alpha=1.0):
    """Return the RGBA code of the given color name with specified transparency."""
    hex_color = PALETTE.get(color_name, '#000000')  # Default to black if color not found
    return hex_to_rgba(hex_color, alpha)

def get_color_list():
    """Return a list of color hex codes from the palette."""
    return list(PALETTE.values())

def visualize_palette():
    """Visualize the color palette."""
    fig, ax = plt.subplots(figsize=(20, 2))
    for i, (color_name, color_hex) in enumerate(PALETTE.items()):
        ax.add_patch(plt.Rectangle((i * 2, 0), 2, 2, color=color_hex))
        ax.text(i * 2 + 1, 1, color_name, color='black' if color_name == 'White' else 'white',
                ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2 * len(PALETTE))
    ax.set_ylim(0, 2)
    ax.axis('off')
    plt.show()

def set_plot_style():
    """Set the default style for plots using LaTeX and custom color palette."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Define the color palette as a list of colors
    color_cycle = [
        PALETTE['Black'],
        PALETTE['Orange'],
        PALETTE['Sky Blue'],
        PALETTE['Bluish Green'],
        PALETTE['Yellow'],
        PALETTE['Blue'],
        PALETTE['Vermillion'],
        PALETTE['Reddish Purple']
    ]

    # Apply Seaborn style and custom settings
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.linewidth': 1,
        'lines.linewidth': 1,
        'lines.markersize': 6,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'pgf.texsystem': 'pdflatex',  # Use pdflatex
        'pgf.rcfonts': False,
        'figure.figsize': (10, 6),  # Default figure size
        'axes.prop_cycle': cycler('color', color_cycle)  # Set the custom color cycle
    })

def set_plot_style_no_latex():
    """Set the default style for plots without requiring LaTeX."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Define the color palette as a list of colors
    color_cycle = [
        PALETTE['Black'],
        PALETTE['Orange'],
        PALETTE['Sky Blue'],
        PALETTE['Bluish Green'],
        PALETTE['Yellow'],
        PALETTE['Blue'],
        PALETTE['Vermillion'],
        PALETTE['Reddish Purple']
    ]

    # Apply Seaborn style and custom settings
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'text.usetex': False,  # Disable LaTeX
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],  # Use a serif font that is similar to LaTeX’s default font
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.linewidth': 1,
        'lines.linewidth': 1,
        'lines.markersize': 6,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.figsize': (10, 6),  # Default figure size
        'axes.prop_cycle': cycler('color', color_cycle)  # Set the custom color cycle
    })

# Optionally, you can also include a function to apply the palette to a plot
def apply_palette(ax, colors):
    """Apply the color palette to a matplotlib axis."""
    for line, color in zip(ax.get_lines(), colors):
        line.set_color(color)
    plt.draw()
