"""Launch napari with the FenestRA dock already open, filling the display.

Opening the dock automatically removes the most common first-run confusion, where the app starts
and the plugin looks like it did not install because nobody found the Plugins menu. Filling the
display matters because the container's X screen is a fixed size: a default-sized napari window
leaves the user looking at a small panel surrounded by black.
"""

import traceback

import napari
from qtpy.QtWidgets import QApplication


def main():
    viewer = napari.Viewer(title="FenestRA")

    try:
        # Public API on napari's Window. availableGeometry, not geometry, so a window manager
        # panel would be respected if one is ever added.
        area = QApplication.primaryScreen().availableGeometry()
        viewer.window.set_geometry(area.x(), area.y(), area.width(), area.height())
    except Exception:
        print("Could not size the window to the display; using napari's default.")
        traceback.print_exc()

    try:
        viewer.window.add_plugin_dock_widget("napari-fenestra", "FenestRA Pipeline")
    except Exception:
        # A napari or npe2 change to this API should not leave the user staring at a blank
        # window with no explanation, so report it and carry on with a usable viewer.
        print("Could not auto-open the FenestRA dock. Open it from Plugins > FenestRA.")
        traceback.print_exc()

    napari.run()


if __name__ == "__main__":
    main()
