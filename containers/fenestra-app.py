"""Launch napari with the FenestRA dock already open, filling the display.

Two things happen here that do not happen in a normal napari launch.

Opening the dock automatically removes the most common first-run confusion, where the app starts
and the plugin looks like it did not install because nobody found the Plugins menu.

Maximising matters because the container's display is not a fixed size: the browser asks Xvnc to
resize the desktop to the browser window, on connect and on every later resize. A *maximised*
window is maintained at the screen size by the window manager, so it follows those changes for
free. An explicitly sized window does not -- measured: with the screen taken from 1920x1080 to
2560x1440, a maximised window moved to 2560x1421 within a second while a window positioned with an
explicit geometry stayed at 1920x1080, leaving a black margin and a small napari in a large desktop.
"""

import traceback

import napari
from qtpy.QtWidgets import QApplication


def main():
    viewer = napari.Viewer(title="FenestRA")

    try:
        # Private attribute, but it is the only route to maximise: napari's Window exposes
        # resize() and set_geometry(), neither of which the window manager will then track.
        viewer.window._qt_window.showMaximized()
    except Exception:
        print("Could not maximise the window; falling back to a fixed size.")
        traceback.print_exc()
        try:
            area = QApplication.primaryScreen().availableGeometry()
            viewer.window.set_geometry(area.x(), area.y(), area.width(), area.height())
        except Exception:
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
