import tkinter as tk
import time

# Initialize the main Tkinter window
root = tk.Tk()
root.title("Depth of Cut GUI")
root.geometry("800x800")  # Set the window size

# Create a frame for displaying the selected side and object
info_frame = tk.Frame(root, bd=2, relief="solid", padx=30, pady=30)
info_frame.pack(pady=10, fill="x")

# Display the selected side
side_label = tk.Label(info_frame, text="Patient Side: N/A", font=("Helvetica", 14), anchor="w")
side_label.pack(fill="x", pady=5)

# Display the selected object
object_label = tk.Label(info_frame, text="Selected Object: N/A", font=("Helvetica", 14), anchor="w")
object_label.pack(fill="x", pady=5)

# Create a frame for displaying marker status
marker_frame = tk.Frame(root, bd=2, relief="solid", padx=10, pady=10)
marker_frame.pack(pady=10, fill="x")

# Create separate labels for Tibia, Pointer, and Verification markers
tibia_label = tk.Label(marker_frame, text="Tibia Marker: N/A", font=("Helvetica", 16), width=30, height=2)
tibia_label.pack(pady=5)

pointer_label = tk.Label(marker_frame, text="Pointer Marker: N/A", font=("Helvetica", 16), width=30, height=2)
pointer_label.pack(pady=5)

verification_label = tk.Label(marker_frame, text="Verification Marker: N/A", font=("Helvetica", 16), width=30, height=2)
verification_label.pack(pady=5)

# Create labels for Coronal, Sagittal, Medial, and Lateral values
coronal_label = tk.Label(root, text="Coronal: N/A", font=("Helvetica", 14), anchor="w")
sagittal_label = tk.Label(root, text="Sagittal: N/A", font=("Helvetica", 14), anchor="w")
medial_label = tk.Label(root, text="Medial: N/A", font=("Helvetica", 14), anchor="w")
lateral_label = tk.Label(root, text="Lateral: N/A", font=("Helvetica", 14), anchor="w")

# Place the labels in the top-right corner
coronal_label.place(x=600, y=20)  # Adjust x and y for positioning
sagittal_label.place(x=600, y=50)
medial_label.place(x=600, y=80)
lateral_label.place(x=600, y=110)


def update_gui():
    """
    Updates the GUI with the current tracking status and instructions.
    """
    global coronal_value, sagittal_value, medial_lateral_value

    # Update the side and object display
    side_label.config(text=f"Patient Side: {CT_side}")
    object_label.config(text=f"Selected Object: {selectedObject}")

    # Update the marker tracking display
    tibia_label.config(text="Tibia Marker", bg="green" if tibia_tracking else "red")
    pointer_label.config(text="Pointer Marker", bg="green" if pointer_tracking else "red")
    verification_label.config(text="Verification Marker", bg="green" if verification_tracking else "red")

    # Update Coronal, Sagittal, and Medial-Lateral values
    coronal_label.config(text=f"Coronal: {varAngle_combined}")
    sagittal_label.config(text=f"Sagittal: {varflex_combined}")
    medial_label.config(text=f"Medial: {Tmc_distance}")
    lateral_label.config(text=f"Lateral: {Tlc_distance}")

    # Schedule the next update
    root.after(100, update_gui)  # Update every 100 milliseconds


# Initialize variables
CT_side = "L"  # Example: Left side
selectedObject = "tibia"  # Example: Tibia selected
tibia_tracking = False
pointer_tracking = False
verification_tracking = False
varAngle_combined = "N/A"
varflex_combined = "N/A"
Tmc_distance = "N/A"
Tlc_distance = "N/A"

# Start the GUI update loop
update_gui()

# Start the Tkinter main loop
root.mainloop()