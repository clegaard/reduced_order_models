import pyvista as pv
import numpy as np

if __name__ == "__main__":

    n_segments_x = 3
    n_segments_y = 3
    segment_size = 1.0
    resolution = 100

    segments = []
    for i in range(n_segments_x):
        for j in range(n_segments_y):

            s = pv.Plane(
                center=(
                    i * segment_size + segment_size / 2,
                    j * segment_size + segment_size / 2,
                    0,
                ),
                i_size=segment_size,
                j_size=segment_size,
                i_resolution=resolution,
                j_resolution=resolution,
            )
            s.cell_data["a"] = np.random.rand() + 0.2
            segments.append(s)

    blocks = pv.MultiBlock(segments)
    blocks.combine()

    # for block in blocks:
    #     surf = block.extract_surface()
    #     tri = surf.triangulate()
    #     tri.plot(show_edges=True)

    # boundary conditions
    # pressure is 1.0 for x=0

    # problem is stationary
    # A u = b
